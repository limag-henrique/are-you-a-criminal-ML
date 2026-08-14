# Research audit v2

Pacote isolado para auditar o conjunto derivado de embeddings sem alterar
`face_profile_ml` nem artefatos históricos. Todos os outputs públicos são
agregados ou pseudonimizados; imagens, nomes, URLs, caminhos pessoais e
embeddings individuais são bloqueados por um scanner fail-closed.

## Comandos de reprodução

Execute a partir da raiz do repositório:

```powershell
# 1. Testes (somente fixtures/controles sintéticos)
python -m pytest -q research_audit_v2

# 2. Integração/reprodutibilidade; não usar para conclusões científicas
python -m research_audit_v2.second_phase.src.run_second_phase --config research_audit_v2/configs/development.yaml

# 3. Configuração científica final; pode ser retomada com segurança
python -m research_audit_v2.second_phase.src.run_second_phase --config research_audit_v2/configs/final.yaml --resume

# 4. Verificação pós-corrida: repete testes, relatórios, hashes e privacidade
python -m research_audit_v2.second_phase.src.final_verification --config research_audit_v2/configs/final.yaml
```

A configuração de desenvolvimento usa no máximo 600 registros, três seeds e
iterações reduzidas. A configuração final usa todos os embeddings disponíveis,
100 seeds explícitas, `k = 32, 48, 64, 80, 96, 128` e `batch_size = 256, 512,
1024, 2048, 4096`.

## Reprodutibilidade e retomada

O manifesto é criado como `initializing` antes do hashing das entradas, passa a
`running` após registrar hashes e só recebe `complete` depois de dois gates de
privacidade — incluindo um scan do manifesto final. Ele registra commit Git,
configuração sem caminhos públicos, seeds, versões, sistema, parâmetros, hashes,
horários, duração, status e hashes dos outputs.

As células de estabilidade usam checkpoints privados em
`research_audit_v2/.checkpoints/`. Um checkpoint só é reutilizado quando os
hashes da entrada, do desenho e dos parâmetros coincidem. Esses arquivos não são
outputs públicos.

## Outputs

- Desenvolvimento: `research_audit_v2/outputs/development/`
- Final: `research_audit_v2/outputs/final/`
- Relatórios editoriais: `research_audit_v2/FINAL_REPRODUCTION_REPORT.md` e
  `research_audit_v2/MANUSCRIPT_UPDATE.md`

Tabelas estruturadas incluem linhagem das cinco contagens, estatísticas de
`group_id`, métricas por dobra, eventos de auditoria de leakage, estabilidade por
execução, resumo P5–P95, comparações pairwise, especificação PCA-64 e controle
sintético.

## Escopo científico

`group_id` significa apenas duplicidade provável, nunca identidade confirmada.
O alvo é reconstruído a partir de clustering e todas as métricas preditivas
medem somente recuperação interna desse alvo sintético. Nenhum resultado mede
criminalidade, culpa, raça/cor, identidade, equidade biométrica ou validade
social, jurídica ou causal.

Resultados históricos preservados, resultados reproduzidos, reconstruções
metodológicas novas e informações não recuperadas são classificados
separadamente. Ausências de fonte, datas ou regras históricas nunca são
preenchidas por inferência.
