# Program corrections

- Added explicit contracts before analysis, preventing mismatched manifest/embedding rows and invalid numerical arrays.
- Added grouped cross-fitting, preventing test observations from fitting clusters, selecting the target or defining the centroid.
- Added public-output privacy scanning and atomic table writes.
- Reduced development clustering iterations from 30 to 8 only for integration feasibility; the locked second-phase design records this and does not treat development output as final.
