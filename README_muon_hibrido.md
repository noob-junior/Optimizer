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
| `rho_alvo_F` | `1.68` | tamanho da perturbação (norma alvo), calibrado |
| `rho_teto_rel` | `0.10` | teto: a perturbação nunca passa de 10% de ‖W‖ |
| `fase2_lr_frac` | `0.1` | LR da fase 2 = `lr × isto` |
| `fase2_patience` | `3` | avaliações sem melhora até encerrar a fase 2 |
| `fase2_teto_frac` | `0.25` | teto da fase 2 = 25% dos passos da fase 1 |
| `gatilho_frac` | `0.15` | dispara a fase 2 quando a melhora cai a 15% do ritmo inicial |

**Se a fase 2 nunca disparar:** aumente `gatilho_frac` (ex.: `0.3`) ou avalie com mais frequência — o gatilho precisa de pelo menos 4 medições de validação.

**Se a fase 2 piorar o resultado:** reduza `rho_alvo_F` (ex.: `0.8`) ou `rho_teto_rel` (ex.: `0.05`). Nada se perde de qualquer forma, porque `best_state()` devolve o melhor ponto observado.

---

## Como a perturbação é dimensionada

Você não escolhe o `ρ` diretamente — escolhe a **norma da perturbação**, e o `ρ` por matriz é derivado dela:

```
‖ε‖_F = ρ · √S · 0,875        S = Σ min(m, n) sobre as matrizes do Muon
```

Isso importa porque **um `ρ` fixo perturba cada vez mais conforme o modelo cresce**. Medimos: com `ρ = 0,03`, um modelo de 3M recebe `‖ε‖ = 1,68` (o ótimo) e um de 31M recebe `3,76` — e já sabíamos que `‖ε‖ ≈ 3` destrói o modelo. Fixando a norma, o `ρ` se ajusta sozinho:

| modelo | ρ automático | dose |
|---|---|---|
| 16 matrizes 256×256 (3M) | 0,0300 | 1,68 |
| 40 matrizes 512×512 (31M) | 0,0134 | 1,68 |

E há um **teto de segurança**: a dose nunca passa de 10% de ‖W‖. Ele só age em modelos pequenos, onde `1,68` seria uma fração alta dos pesos:

| modelo | dose sem teto | dose com teto |
|---|---|---|
| MLP pequeno | 1,68 (26% de ‖W‖) | **0,47** (10%) |
| CNN pequena | 1,68 (38% de ‖W‖) | **0,44** (10%) |
| Transformer 3M | 1,68 (5% de ‖W‖) | 1,68 (intacto) |

Confira o que o seu modelo recebeu:

```python
print(opt.resumo())
# Muon-SR | 16 matrizes via Muon | S=4096 | ρ_eff=0.0300 (‖ε‖F alvo=1.68) | ...
```

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

**`RuntimeError` no meio de um treino longo**
Se a fase 2 disparar e você não estiver passando `closure`, o erro aparece só naquele momento — depois de horas de treino. Passe o `closure` desde o início.

---

## Limitações conhecidas

**O estado das fases não entra no `state_dict()`.** Se você salvar e recarregar o otimizador no meio do treino, a fase, o melhor checkpoint e os contadores de passo se perdem — o treino recomeça na fase 1. Para retomar treinos longos, salve também:

```python
extra = dict(fase=opt.fase, best_val=opt.best_val, passos_f1=opt.passos_f1)
```

**A perturbação só age nas matrizes do Muon.** Os parâmetros do AdamW (embeddings, saída, normalizações) recebem gradientes calculados no ponto perturbado, mas não são perturbados eles próprios. É intencional — a geometria espectral não faz sentido para eles.

**A dose foi calibrada em um transformer.** Ver a nota no fim deste guia.

---

## Uma nota de honestidade

A dose `‖ε‖ = 1.68` foi **medida** num transformer de 3M, com grade fina e 20 sementes pareadas. Já a regra `ρ ∝ 1/√S` e o teto de 10% são **derivados** — explicam uma falha que observamos num modelo de 31M, mas não foram validados diretamente.

Num teste em 31M com `ρ` transferido sem a correção, a perturbação espectral **perdeu** para a euclidiana (−0,0097, 4 sementes, intervalo ainda cruzando o zero). A explicação mais provável é super-dose: aquele `ρ` entregava `‖ε‖ = 3,76`. É essa falha que a fórmula corrige.

O que sabemos com segurança é que **a fase 2 rende ~+0,11 nas duas escalas** — o refinamento em si transfere bem. Qual geometria é melhor é que depende da escala, e vale ~5%.

Se tiver tempo antes do benchmark, varra `rho_alvo_F ∈ {0.8, 1.68, 3.0}` numa das suas arquiteturas. É o número que mais vale confirmar no lugar onde vai ser usado.
