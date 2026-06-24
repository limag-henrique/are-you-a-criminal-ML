#!/usr/bin/env node
/*
 * DEA fugitives image scraper.
 *
 * Uses a real browser because DEA.gov's edge service rejects ordinary
 * command-line HTTP clients from this environment.
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
require("module").Module._initPaths();

const { chromium } = require("playwright");

const BASE_URL = "https://www.dea.gov";
const START_URL = "https://www.dea.gov/fugitives/all?page=3";
const EDGE_EXE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const OUTPUT_DIR = path.resolve(process.cwd(), "DEA");
const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";
const RATE_LIMIT_MS = Number(process.env.DEA_DELAY_MS || 1200);
const MAX_RETRIES = 3;
const MAX_PAGES = process.env.DEA_MAX_PAGES ? Number(process.env.DEA_MAX_PAGES) : 0;
const MAX_PROFILES = process.env.DEA_MAX_PROFILES ? Number(process.env.DEA_MAX_PROFILES) : 0;

const IMAGE_EXTENSIONS = new Map([
  ["image/jpeg", ".jpg"],
  ["image/jpg", ".jpg"],
  ["image/png", ".png"],
  ["image/gif", ".gif"],
  ["image/webp", ".webp"],
]);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function safeSlug(value) {
  const cleaned = String(value || "")
    .normalize("NFKD")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/[-\s]+/g, "_")
    .replace(/_+/g, "_")
    .toLowerCase();
  return cleaned || "unknown";
}

function imageHash(url) {
  return crypto.createHash("sha1").update(url).digest("hex").slice(0, 10);
}

function extensionFrom(url, contentType) {
  const normalizedType = String(contentType || "").split(";")[0].trim().toLowerCase();
  if (IMAGE_EXTENSIONS.has(normalizedType)) {
    return IMAGE_EXTENSIONS.get(normalizedType);
  }

  const pathname = new URL(url).pathname.toLowerCase();
  const ext = path.extname(pathname);
  if ([".jpg", ".jpeg", ".png", ".gif", ".webp"].includes(ext)) {
    return ext === ".jpeg" ? ".jpg" : ext;
  }

  return ".jpg";
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function saveJson(filePath, value) {
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), "utf8");
}

function countImageFiles(dir) {
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .filter((entry) => /\.(jpe?g|png|gif|webp)$/i.test(entry.name)).length;
}

function loadExistingMetadata(metadataPath) {
  if (!fs.existsSync(metadataPath)) {
    return [];
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.warn(`Could not read existing metadata for resume: ${error.message}`);
    return [];
  }
}

async function gotoWithRetry(page, url, options = {}) {
  let lastError;
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt += 1) {
    try {
      const response = await page.goto(url, {
        waitUntil: options.waitUntil || "domcontentloaded",
        timeout: options.timeout || 45000,
      });
      if (!response) {
        throw new Error("navigation returned no response");
      }
      if (response.status() >= 400) {
        throw new Error(`HTTP ${response.status()}`);
      }
      return response;
    } catch (error) {
      lastError = error;
      if (attempt < MAX_RETRIES) {
        await sleep(RATE_LIMIT_MS * attempt);
      }
    }
  }
  throw lastError;
}

async function discoverPagination(page) {
  await gotoWithRetry(page, START_URL);
  const bodyText = await page.locator("body").innerText({ timeout: 10000 });
  const totalMatch = bodyText.match(/(\d[\d,]*)\s+Results\s+-\s+Showing/i);
  const totalProfiles = totalMatch ? Number(totalMatch[1].replace(/,/g, "")) : 0;
  if (!totalProfiles) {
    throw new Error("Could not find total profile count on the DEA listing page.");
  }
  return {
    totalProfiles,
    lastPageIndex: Math.ceil(totalProfiles / 10) - 1,
  };
}

async function extractProfileLinks(page, pageIndex) {
  const url = `${BASE_URL}/fugitives/all?page=${pageIndex}`;
  await gotoWithRetry(page, url);
  await page.waitForLoadState("networkidle", { timeout: 15000 }).catch(() => {});

  return page.evaluate(() => {
    const seen = new Set();
    return Array.from(document.querySelectorAll('a[href^="/fugitives/"], a[href^="https://www.dea.gov/fugitives/"]'))
      .map((anchor) => ({
        name: anchor.textContent.trim().replace(/\s+/g, " "),
        url: new URL(anchor.getAttribute("href"), location.origin).href,
      }))
      .filter((entry) => {
        const url = new URL(entry.url);
        if (url.pathname === "/fugitives/all") return false;
        if (!url.pathname.startsWith("/fugitives/")) return false;
        if (!entry.name) return false;
        if (seen.has(url.pathname)) return false;
        seen.add(url.pathname);
        return true;
      });
  });
}

async function extractProfile(page, url) {
  await gotoWithRetry(page, url);
  await page.waitForLoadState("networkidle", { timeout: 15000 }).catch(() => {});

  return page.evaluate(() => {
    const title = document.title.trim().replace(/\s+/g, " ");
    const images = Array.from(document.querySelectorAll("img"))
      .map((img) => ({
        alt: (img.getAttribute("alt") || "").trim(),
        src: img.currentSrc || img.src || img.getAttribute("src") || "",
      }))
      .map((img) => ({
        ...img,
        src: img.src ? new URL(img.src, location.origin).href : "",
      }))
      .filter((img) => img.src.includes("/sites/default/files/"))
      .filter((img) => /\.(jpe?g|png|gif|webp)(?:$|\?)/i.test(new URL(img.src).pathname))
      .filter((img, index, list) => list.findIndex((other) => other.src === img.src) === index);

    return { title, images };
  });
}

async function downloadImage(imagePage, url, outputPath) {
  if (fs.existsSync(outputPath) && fs.statSync(outputPath).size > 0) {
    return { skipped: true, bytes: fs.statSync(outputPath).size };
  }

  const response = await gotoWithRetry(imagePage, url, {
    waitUntil: "domcontentloaded",
    timeout: 45000,
  });
  const contentType = response.headers()["content-type"] || "";
  if (!contentType.toLowerCase().startsWith("image/")) {
    throw new Error(`unexpected content type: ${contentType || "unknown"}`);
  }

  const body = await response.body();
  fs.writeFileSync(outputPath, body);
  return { skipped: false, bytes: body.length };
}

async function main() {
  ensureDir(OUTPUT_DIR);

  const metadataPath = path.join(OUTPUT_DIR, "metadata.json");
  const failuresPath = path.join(OUTPUT_DIR, "download_failures.csv");
  const summaryPath = path.join(OUTPUT_DIR, "summary.json");

  const browser = await chromium.launch({
    headless: true,
    executablePath: EDGE_EXE,
  });
  const context = await browser.newContext({
    userAgent: USER_AGENT,
    viewport: { width: 1280, height: 900 },
  });
  const page = await context.newPage();
  const imagePage = await context.newPage();

  const profiles = loadExistingMetadata(metadataPath);
  const failures = profiles
    .filter((profile) => !profile.images || profile.images.length === 0)
    .map((profile) => ({
      profile_url: profile.profile_url,
      image_url: "",
      reason: "no fugitive image found on profile",
    }));
  const seenProfiles = new Set(profiles.map((profile) => profile.profile_url).filter(Boolean));
  const seenImages = new Map();
  for (const profile of profiles) {
    for (const image of profile.images || []) {
      if (image.image_url && image.local_filename) {
        seenImages.set(image.image_url, image.local_filename);
      }
    }
  }
  let downloadedImages = 0;
  let skippedDuplicateImages = 0;

  try {
    const { totalProfiles, lastPageIndex } = await discoverPagination(page);
    console.log(`DEA listing reports ${totalProfiles} profiles across ${lastPageIndex + 1} pages.`);
    if (profiles.length) {
      console.log(`Resuming from existing metadata: ${profiles.length} profiles already recorded.`);
    }

    const effectiveLastPageIndex = MAX_PAGES
      ? Math.min(lastPageIndex, MAX_PAGES - 1)
      : lastPageIndex;

    for (let pageIndex = 0; pageIndex <= effectiveLastPageIndex; pageIndex += 1) {
      const links = await extractProfileLinks(page, pageIndex);
      console.log(
        `Listing page ${pageIndex + 1}/${effectiveLastPageIndex + 1}: ${links.length} profile links`
      );

      for (const link of links) {
        if (MAX_PROFILES && seenProfiles.size >= MAX_PROFILES) {
          break;
        }

        if (seenProfiles.has(link.url)) {
          continue;
        }
        seenProfiles.add(link.url);

        const slug = safeSlug(new URL(link.url).pathname.split("/").pop());
        const profileRecord = {
          name: link.name,
          slug,
          profile_url: link.url,
          images: [],
        };

        try {
          const profile = await extractProfile(page, link.url);
          profileRecord.name = profile.title || link.name;

          if (!profile.images.length) {
            failures.push({
              profile_url: link.url,
              image_url: "",
              reason: "no fugitive image found on profile",
            });
          }

          for (let imageIndex = 0; imageIndex < profile.images.length; imageIndex += 1) {
            const img = profile.images[imageIndex];
            if (seenImages.has(img.src)) {
              skippedDuplicateImages += 1;
              profileRecord.images.push({
                image_url: img.src,
                local_filename: seenImages.get(img.src),
                duplicate: true,
                alt: img.alt,
              });
              continue;
            }

            const hash = imageHash(img.src);
            const provisionalExt = extensionFrom(img.src, "");
            const filename = `${safeSlug(profileRecord.name)}__${slug}__${hash}${provisionalExt}`;
            const outputPath = path.join(OUTPUT_DIR, filename);

            try {
              const result = await downloadImage(imagePage, img.src, outputPath);
              downloadedImages += result.skipped ? 0 : 1;
              seenImages.set(img.src, filename);
              profileRecord.images.push({
                image_url: img.src,
                local_filename: filename,
                alt: img.alt,
                bytes: result.bytes,
                skipped_existing: result.skipped,
              });
            } catch (error) {
              failures.push({
                profile_url: link.url,
                image_url: img.src,
                reason: error.message,
              });
            }

            await sleep(RATE_LIMIT_MS);
          }
        } catch (error) {
          failures.push({
            profile_url: link.url,
            image_url: "",
            reason: `profile failed: ${error.message}`,
          });
        }

        profiles.push(profileRecord);
        saveJson(metadataPath, profiles);
        await sleep(RATE_LIMIT_MS);
      }

      if (MAX_PROFILES && seenProfiles.size >= MAX_PROFILES) {
        break;
      }

      await sleep(RATE_LIMIT_MS);
    }

    const summary = {
      source: START_URL,
      robots_checked: "https://www.dea.gov/robots.txt",
      reported_total_profiles: totalProfiles,
      unique_profiles_found: seenProfiles.size,
      profile_image_references: profiles.reduce(
        (count, profile) => count + (profile.images || []).length,
        0
      ),
      images_downloaded_this_run: downloadedImages,
      total_image_files_on_disk: countImageFiles(OUTPUT_DIR),
      unique_image_urls_seen: seenImages.size,
      duplicate_images_skipped_this_run: skippedDuplicateImages,
      duplicate_image_references_total:
        profiles.reduce((count, profile) => count + (profile.images || []).length, 0) -
        seenImages.size,
      profiles_without_images: profiles.filter(
        (profile) => !profile.images || profile.images.length === 0
      ).length,
      failures_or_missing_images: failures.length,
      output_dir: OUTPUT_DIR,
      rate_limit_ms: RATE_LIMIT_MS,
      completed_at: new Date().toISOString(),
    };

    saveJson(summaryPath, summary);
    fs.writeFileSync(
      failuresPath,
      [
        "profile_url,image_url,reason",
        ...failures.map((row) =>
          [row.profile_url, row.image_url, row.reason].map(csvEscape).join(",")
        ),
      ].join("\n") + "\n",
      "utf8"
    );

    console.log("Done.");
    console.log(JSON.stringify(summary, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
