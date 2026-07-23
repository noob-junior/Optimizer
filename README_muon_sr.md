# Muon-SR — guia de uso

Otimizador de duas fases:

- **Fase 1** — Muon nas matrizes ocultas + AdamW no resto. Uma passada por passo.
- **Fase 2** — a mesma base Muon, com uma perturbação na geometria espectral ligada. Duas passadas por passo.

A troca de fase é **automática**. Você não decide quando ela acontece.

---

## Instalação

Só precisa de PyTorch. Copie `muon_sr_simples.py` para o seu projeto:

```python
from muon_sr_simples import MuonSR
```

Verifique se está tudo certo antes de usar:

```bash
python muon_sr_simples.py            # roda as auditorias
python muon_sr_simples.py --exemplo  # mostra o treino com a troca de fase
```

---

## Uso básico

São três coisas: criar, dar o passo com `closure`, e avisar da validação.

```python
from muon_sr_simples import MuonSR

model = MinhaRede()
opt = MuonSR(model, lr=3e-3)          # passe o MODELO, não model.parameters()

for passo in range(1, N+1):
    xb, yb = proximo_batch()

    # 1ª passada (normal)
    opt.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward()

    # 2ª passada — usada só quando a fase 2 estiver ativa
    def closure():
        opt.zero_grad()
        l = criterion(model(xb), yb)
        l.backward()
        return l

    opt.step(closure)

    # avise o otimizador a cada avaliação
    if passo % 100 == 0:
        val = avaliar(model)
        st = opt.observe_val(val)
        if st['fase'] == 3:
            break                      # fase 2 terminou; nada mais a ganhar

# entregue o MELHOR ponto, não o último
model.load_state_dict(opt.best_state())
```

### Por que passar o modelo, e não os parâmetros

O otimizador precisa dos **nomes** dos parâmetros para decidir o que vai para o Muon e o que vai para o AdamW, e precisa do `state_dict()` para guardar o melhor checkpoint.

### Por que o `closure`

O SAM precisa de duas passadas: uma para achar a direção da perturbação, outra para medir o gradiente no ponto perturbado. O `closure` é a segunda.

Na fase 1 ele é ignorado — **você pode sempre passá-lo**, sem custo.

---

## Os três métodos

| método | quando chamar | o que faz |
|---|---|---|
| `opt.step(closure)` | todo passo | aplica a atualização |
| `opt.observe_val(val)` | a cada avaliação | controla a troca de fase e guarda o melhor |
| `opt.best_state()` | no fim | devolve os pesos do melhor ponto visto |

**`observe_val` não é opcional.** Sem ele o otimizador nunca sai da fase 1 e nunca guarda o melhor checkpoint.

O retorno de `observe_val` serve para logar:

```python
st = opt.observe_val(val)
# {'fase': 1, 'val': 0.53, 'best_val': 0.53, 'best_fase': 1, 'rho': 0.0, 'passos': 400}
```

`fase` vale `1` (Muon), `2` (refinando) ou `3` (terminou).

---

## Lendo o log

```
passo  300 | fase 1 | val 1.69183 | best 1.69183 (fase 1) | ρ=0.0000
passo  400 | fase 2 | val 0.70689 | best 0.70689 (fase 1) | ρ=0.0000   ← trocou de fase
passo  425 | fase 2 | val 0.69152 | best 0.69152 (fase 2) | ρ=0.0305   ← ρ subindo
passo  450 | fase 2 | val 0.68873 | best 0.68873 (fase 2) | ρ=0.0609
passo  500 | fase 3 | val 0.69687 | best 0.68873 (fase 2) | ρ=0.0000   ← encerrou
```

O `ρ` sobe de zero devagar de propósito: assim a curva de validação não dá um solavanco na troca de fase.

---

## Hiperparâmetros

Na maioria dos casos você só mexe no `lr`.

| parâmetro | padrão | o que faz |
|---|---|---|
| `lr` | `3e-3` | taxa de aprendizado da fase 1 |
| `momentum` | `0.95` | momentum do Muon |
| `weight_decay` | `0.01` | decaimento de peso |
| `rho_rel` | `0.20` | tamanho da perturbação, como fração de ‖W‖ |
| `fase2_lr_frac` | `0.1` | LR da fase 2 = `lr × isto` |
| `fase2_patience` | `3` | avaliações sem melhora até encerrar a fase 2 |
| `fase2_teto_frac` | `0.25` | teto da fase 2 = 25% dos passos da fase 1 |
| `gatilho_frac` | `0.15` | dispara a fase 2 quando a melhora cai a 15% do ritmo inicial |

**Se a fase 2 nunca disparar:** aumente `gatilho_frac` (ex.: `0.3`) ou avalie com mais frequência — o gatilho precisa de pelo menos 4 medições de validação.

**Se a fase 2 piorar o resultado:** reduza `rho_rel` para `0.1`. Nada se perde de qualquer forma, porque `best_state()` devolve o melhor ponto observado.

---

## Arquiteturas

Funciona em MLP, CNN e Transformer sem mudar nada. O split é automático:

| parâmetro | otimizador |
|---|---|
| matrizes ocultas (`Linear`, atenção, MLP) | Muon |
| kernels de convolução | Muon (remodelados) |
| embeddings, camada de saída, normalizações, vieses | AdamW |

Confira o split antes de treinar:

```python
print(opt.resumo())
# Muon-SR | 4 matrizes via Muon | S=128 | ρ_eff=0.1745 | fase=1 | ...
```

Se as suas camadas de embedding ou saída usam nomes fora do comum, adicione-os à lista `_EXCLUIR` no topo do arquivo — é o único ponto que depende de convenção de nomes.

**Sobre CNNs:** o Muon foi feito para matrizes de operador linear, e a literatura mede ganho grande em transformers e modesto em redes convolucionais. Funciona, mas espere menos.

---

## Erros comuns

**`RuntimeError: A fase 2 precisa de closure`**
Você chamou `opt.step()` sem `closure` depois de a fase 2 começar. Passe sempre o `closure`.

**A fase 2 nunca acontece**
Você não está chamando `observe_val`, ou está avaliando poucas vezes (precisa de ≥4 medições).

**`best_state()` retorna `None`**
`observe_val` nunca foi chamado.

**O `closure` não pode esquecer o `zero_grad()`**
Sem ele, os gradientes das duas passadas se somam e o passo sai errado.

---

## Uma nota de honestidade

O valor `rho_rel = 0.20` é uma **estimativa derivada**, não medido diretamente. A calibração original (`‖ε‖ = 1.68`) foi feita num transformer pequeno, com 20 sementes pareadas; a conversão para fração de ‖W‖ é uma inferência.

Se tiver tempo, vale varrer `rho_rel ∈ {0.1, 0.2, 0.4}` numa das suas arquiteturas antes de rodar o benchmark. É o único número do desenho que ainda não foi medido no lugar onde vai ser usado.
