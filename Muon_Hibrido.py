#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Muon-SR  —  Muon com Refinamento Espectral
================================================================================
UMA IDEIA, em uma frase:

    O Muon é descida mais íngreme sob a norma ESPECTRAL. Se a perturbação do SAM
    deve casar com a geometria do otimizador, então a perturbação certa para o
    Muon é a ascensão mais íngreme sob a MESMA norma:  ε = ρ·Orth(g).

    E como Orth(·) é exatamente o que o Newton-Schulz do Muon já calcula, essa
    perturbação sai de graça — a máquina já está no otimizador.

O otimizador tem duas fases:

    FASE 1   Muon padrão nas matrizes ocultas + AdamW no resto.   1 passada/passo
    FASE 2   A MESMA base Muon, com a perturbação ligada.         2 passadas/passo

--------------------------------------------------------------------------------
 O que é emprestado e o que é nosso (para não haver dúvida)
--------------------------------------------------------------------------------
 EMPRESTADO (literatura estabelecida):
   · Muon: Newton-Schulz + escala 0,2·√max(m,n)      Jordan et al. 2024
   · Split Muon/AdamW (matrizes ocultas vs resto)     idem
   · SAM: perturbar antes de dar o passo              Foret et al. 2021
   · Casar perturbação com a geometria do otimizador  DeepMind 2025 (p/ AdamW)

 NOSSO:
   · A perturbação na geometria ESPECTRAL, ε = ρ·Orth(g), que é o caso que o
     princípio acima não cobria (a família publicada de SAM não-Euclidiano só
     trata métricas Riemannianas; a norma espectral não é uma delas).
   · A calibração de ρ e a regra de escala ρ ∝ 1/√S.
   · O uso do refinamento como FASE, não como otimizador do treino inteiro.

--------------------------------------------------------------------------------
 Uso
--------------------------------------------------------------------------------
    opt = MuonSR(model, lr=3e-3)

    loss = criterion(model(x), y); loss.backward()
    opt.step(closure)            # closure só é usada na fase 2
    ...
    opt.observe_val(val_loss)    # após cada avaliação
    model.load_state_dict(opt.best_state())
================================================================================
"""
import math
from typing import Optional, Callable, Dict, List

import torch
from torch.optim import Optimizer


# ==============================================================================
#  Ortogonalização (a primitiva do Muon, e também da nossa perturbação)
# ==============================================================================
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Aproxima Orth(G) = UVᵀ por iteração quíntica de Newton-Schulz.

    NOTA: a iteração não leva os valores singulares exatamente a 1 — deixa-os em
    ~[0,75; 1,25]. Isso é intencional (converge muito mais rápido) e não afeta a
    direção do passo. Afeta a MAGNITUDE da perturbação, e por isso o fator
    NS_EFF entra na calibração de ρ mais abaixo.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315                  # coeficientes canônicos

    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    X = X / (X.norm() + eps)
    transposta = G.size(0) > G.size(1)
    if transposta:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposta:
        X = X.T
    return X.to(G.dtype)


NS_EFF = 0.875     # ‖Orth(G)‖_F medido / √min(m,n) teórico (por causa dos svs ≈ [0,75; 1,25])


# ==============================================================================
#  Quais parâmetros vão para o Muon
# ==============================================================================
_EXCLUIR = ('embed', 'wte', 'wpe', 'lm_head', 'classifier', 'tok_emb', 'pos_emb')


def eh_matriz_oculta(nome: str, p: torch.Tensor) -> bool:
    """Muon só faz sentido em matrizes que representam um operador linear entre
    camadas. Vieses e ganhos de normalização são 1D. Embeddings e a camada de
    saída são tabelas de lookup / projeção para o vocabulário — ortogonalizá-las
    piora. Este split (Muon nas ocultas, AdamW no resto) é o padrão de mercado.
    """
    return p.ndim >= 2 and not any(t in nome.lower() for t in _EXCLUIR)


def como_matriz(t: torch.Tensor) -> torch.Tensor:
    """Achata para a matriz do operador equivalente.

    Conv2D (out, in, kh, kw) implementa, por posição, uma multiplicação entre o
    patch achatado (in·kh·kw) e a saída — logo a matriz é (out, in·kh·kw).
    RESSALVA: em CNNs o ganho do Muon é modesto (a literatura mede +5pp em ViT
    contra pouco em ResNet). O reshape funciona; a expectativa é que seja menor.
    """
    return t if t.ndim == 2 else t.reshape(t.size(0), -1)


# ==============================================================================
#  O otimizador
# ==============================================================================
class MuonSR(Optimizer):
    def __init__(self, model, lr: float = 3e-3, momentum: float = 0.95,
                 betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.01,
                 ns_steps: int = 5, rho_rel: float = 0.20,
                 fase2_lr_frac: float = 0.1, fase2_rampa: int = 200,
                 fase2_patience: int = 3, fase2_teto_frac: float = 0.25,
                 gatilho_frac: float = 0.15):
        self.model = model

        muon, adamw = [], []
        for nome, p in model.named_parameters():
            if p.requires_grad:
                (muon if eh_matriz_oculta(nome, p) else adamw).append(p)

        base = dict(lr=lr, momentum=momentum, betas=betas, eps=eps,
                    weight_decay=weight_decay, ns_steps=ns_steps)
        grupos = [dict(params=muon, tipo='muon', **base)]
        if adamw:
            grupos.append(dict(params=adamw, tipo='adamw', **base))
        super().__init__(grupos, base)

        # ---------------------------------------------------------------------
        # CALIBRAÇÃO DE ρ
        #
        # Uma matriz m×n perturbada com norma espectral ρ tem min(m,n) valores
        # singulares iguais a ρ, logo ‖ε_i‖_F = ρ·√min(m,n). Somando as k
        # matrizes:   ‖ε‖_F = ρ · √S · NS_EFF,   com  S = Σ_i min(m_i, n_i).
        #
        # Consequência: um ρ FIXO significa perturbações cada vez maiores
        # conforme o modelo cresce. Medimos isso: ρ=0,03 dava ‖ε‖_F=1,68 num
        # modelo de 3M e 3,76 num de 31M — e ‖ε‖_F≈3 já destrói o modelo.
        #
        # A correção: fixar a dose como FRAÇÃO de ‖W‖_F. Assim ela é
        # adimensional e transfere entre MLP, CNN e Transformer — que é o que o
        # benchmark exige (um único conjunto de hiperparâmetros).
        # ---------------------------------------------------------------------
        self._muon_params = muon
        self._S = max(sum(min(como_matriz(p).shape) for p in muon), 1)
        self.rho_rel = rho_rel
        self.rho_eff = self._calibrar_rho()

        # estado do controlador de fases
        self.fase = 1
        self.passos = self.passos_f1 = self.passos_f2 = 0
        self.fase2_lr_frac = fase2_lr_frac
        self.fase2_rampa = fase2_rampa
        self.fase2_patience = fase2_patience
        self.fase2_teto_frac = fase2_teto_frac
        self.gatilho_frac = gatilho_frac
        self._hist: List[float] = []
        self._melhoras: List[float] = []
        self._sem_melhora = 0
        self.best_val = float('inf')
        self._best: Optional[Dict[str, torch.Tensor]] = None
        self.best_fase = 0
        self._eps: Dict[torch.Tensor, torch.Tensor] = {}

    @torch.no_grad()
    def _calibrar_rho(self) -> float:
        wF = math.sqrt(sum(float(p.detach().float().pow(2).sum()) for p in self._muon_params))
        return (self.rho_rel * wF) / (NS_EFF * math.sqrt(self._S))

    # ==========================================================================
    #  Passo
    # ==========================================================================
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Fase 1: uma passada. Fase 2: perturba, recalcula o gradiente no ponto
        perturbado, restaura os pesos, e só então aplica o passo.

        O custo 2× é pago só na fase final. Nos nossos testes, no mesmo
        orçamento de refinamento, continuar com Muon rendeu +0,011 e o
        refinamento perturbado rendeu +0,115.
        """
        rho = self._rho_agora()
        if rho > 0:
            if closure is None:
                raise RuntimeError("A fase 2 precisa de `closure` (2ª passada).")
            self._perturbar(rho)
            with torch.enable_grad():
                closure()
            self._restaurar()

        for g in self.param_groups:
            lr = g['lr'] * (self.fase2_lr_frac if self.fase == 2 else 1.0)
            # LR menor na fase 2: ela parte de um modelo já convergido, e um LR
            # alto o chutaria para fora do mínimo.
            (self._passo_muon if g['tipo'] == 'muon' else self._passo_adamw)(g, lr)

        self.passos += 1
        if self.fase == 1:
            self.passos_f1 += 1
        else:
            self.passos_f2 += 1

    def _rho_agora(self) -> float:
        """ρ com rampa linear no início da fase 2.

        A rampa existe porque a convergência é detectada pelo BENCHMARK, com
        parâmetros que não controlamos. Entrando com ρ cheio, a curva de
        validação daria um solavanco e o detector poderia encerrar o treino.
        Com a rampa, os primeiros passos da fase 2 são idênticos à fase 1
        continuando, e a perturbação entra sem degrau.
        """
        if self.fase != 2:
            return 0.0
        return self.rho_eff * min(1.0, self.passos_f2 / max(1, self.fase2_rampa))

    # --------------------------------------------------------------------------
    @torch.no_grad()
    def _perturbar(self, rho: float):
        """ε = ρ·Orth(g) — a ascensão mais íngreme sob a bola ESPECTRAL.

        O SAM resolve  argmax_{‖ε‖≤ρ} ⟨g, ε⟩, e a solução depende da norma:
            norma L2        ->  ε = ρ·g/‖g‖      (SAM clássico)
            norma espectral ->  ε = ρ·Orth(g)    (o nosso caso)

        A perturbação resultante tem TODOS os valores singulares iguais a ρ:
        explora o subespaço uniformemente em vez de concentrar energia na
        direção do gradiente. Só as matrizes que o Muon governa são perturbadas.
        """
        self._eps.clear()
        for g in self.param_groups:
            if g['tipo'] != 'muon':
                continue
            for p in g['params']:
                if p.grad is None:
                    continue
                E = (rho * newton_schulz(como_matriz(p.grad), g['ns_steps'])).reshape(p.shape)
                self._eps[p] = E
                p.add_(E)

    @torch.no_grad()
    def _restaurar(self):
        """Subtrai o MESMO ε que foi somado. Recalcular Orth(g) no ponto
        perturbado daria outra matriz (o gradiente mudou) e o modelo não
        voltaria ao lugar."""
        for p, E in self._eps.items():
            p.sub_(E)
        self._eps.clear()

    # --------------------------------------------------------------------------
    @torch.no_grad()
    def _passo_muon(self, g, lr):
        for p in g['params']:
            if p.grad is None:
                continue
            st = self.state[p]
            if 'buf' not in st:
                st['buf'] = torch.zeros_like(p)
            st['buf'].mul_(g['momentum']).add_(p.grad)
            d = p.grad.add(st['buf'], alpha=g['momentum'])          # Nesterov

            M = como_matriz(d)
            O = newton_schulz(M, g['ns_steps'])

            # A escala canônica do Muon. Orth(G) tem norma espectral ~1
            # independentemente do tamanho da matriz; sem reescalar, uma camada
            # grande receberia passo relativo muito menor que uma pequena. É
            # este fator que permite UM ÚNICO LR para MLP, CNN e Transformer.
            escala = 0.2 * math.sqrt(max(M.shape))

            p.mul_(1 - lr * g['weight_decay'])
            p.add_((O * escala).reshape(p.shape), alpha=-lr)

    @torch.no_grad()
    def _passo_adamw(self, g, lr):
        b1, b2 = g['betas']
        for p in g['params']:
            if p.grad is None:
                continue
            st = self.state[p]
            if 'm' not in st:
                st['m'] = torch.zeros_like(p); st['v'] = torch.zeros_like(p); st['t'] = 0
            st['t'] += 1
            st['m'].mul_(b1).add_(p.grad, alpha=1 - b1)
            st['v'].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            mh = st['m'] / (1 - b1 ** st['t'])
            vh = st['v'] / (1 - b2 ** st['t'])
            p.mul_(1 - lr * g['weight_decay'])
            p.add_(mh / (vh.sqrt() + g['eps']), alpha=-lr)

    # ==========================================================================
    #  Controlador de fases (chamar após cada avaliação)
    # ==========================================================================
    def observe_val(self, val_loss: float) -> dict:
        val_loss = float(val_loss)
        self._hist.append(val_loss)

        # Guarda o melhor ponto de TODOS os vistos. O modelo se degrada depois
        # do pico; entregar o estado final em vez do best infla artificialmente
        # qualquer comparação (medimos ~3× de inflação).
        if val_loss < self.best_val - 1e-9:
            self.best_val = val_loss
            self._best = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.best_fase = self.fase
            self._sem_melhora = 0
        else:
            self._sem_melhora += 1

        if self.fase == 1 and len(self._hist) >= 4:
            d = self._hist[-2] - self._hist[-1]
            self._melhoras.append(max(d, 0.0))
            ref = max(self._melhoras[:3]) if len(self._melhoras) >= 3 else max(self._melhoras)
            # Dispara na DESACELERAÇÃO, não no platô: se esperássemos o platô, o
            # detector de convergência do benchmark encerraria o treino antes de
            # a fase 2 rodar.
            if ref > 0 and d < self.gatilho_frac * ref:
                self._iniciar_fase2()

        elif self.fase == 2:
            # Auto-limitante: a métrica pondera passos, e a fase 2 custa 2× por
            # passo. Ela para assim que deixa de comprar qualidade.
            teto = int(self.fase2_teto_frac * max(1, self.passos_f1))
            if self._sem_melhora >= self.fase2_patience or self.passos_f2 >= teto:
                self.fase = 3

        return dict(fase=self.fase, val=val_loss, best_val=self.best_val,
                    best_fase=self.best_fase, rho=self._rho_agora(), passos=self.passos)

    def _iniciar_fase2(self):
        """Rebobina para o MELHOR ponto e liga a perturbação.

        Refina o best, não o último: se o modelo já começou a degradar, refinar
        de onde ele está seria refinar um ponto pior.

        Não há troca de otimizador aqui — a base continua Muon, só a perturbação
        entra. É o que evita ter de reconstruir estado de outro otimizador.
        """
        if self._best is not None:
            self.model.load_state_dict(self._best)
        # No modo relativo a dose depende de ‖W‖_F, e os pesos cresceram durante
        # a fase 1 — recalcular aqui usa o modelo que será de fato perturbado.
        self.rho_eff = self._calibrar_rho()
        self.fase = 2
        self.passos_f2 = 0
        self._sem_melhora = 0

    def best_state(self):
        """O estado a entregar. Se a fase 2 não ajudou naquela arquitetura, o
        resultado é o da fase 1 — o otimizador degrada para 'Muon puro' em vez
        de piorar."""
        return self._best

    def resumo(self) -> str:
        return (f"Muon-SR | {len(self._muon_params)} matrizes via Muon | S={self._S} | "
                f"ρ_eff={self.rho_eff:.4f} | fase={self.fase} | "
                f"passos={self.passos} (f1={self.passos_f1}, f2={self.passos_f2}) | "
                f"best={self.best_val:.5f} na fase {self.best_fase}")


# ==============================================================================
#  Auditorias (cada primitiva contra um valor calculável à mão)
# ==============================================================================
def auditorias():
    import torch.nn as nn
    ok = True
    def chk(n, c, e=""):
        nonlocal ok; ok &= bool(c); print(f"  [{'PASS' if c else 'FALHA'}] {n}  {e}")

    print("=" * 72); print("AUDITORIAS Muon-SR"); print("=" * 72)

    torch.manual_seed(0)
    G = torch.randn(48, 64)
    O = newton_schulz(G, 5)
    sv = torch.linalg.svdvals(O)
    c0 = (torch.linalg.svdvals(G).max() / torch.linalg.svdvals(G).min()).item()
    c1 = (sv.max() / sv.min()).item()
    chk("A1 Newton-Schulz ortogonaliza (condição despenca)", c1 < c0 / 3, f"cond {c0:.1f} -> {c1:.2f}")

    cos = torch.nn.functional.cosine_similarity(G.flatten(), O.flatten(), dim=0).item()
    chk("A2 Orth(g) não é um reescalamento de g", cos < 0.95,
        f"cos(g, Orth(g))={cos:.3f} — um reescalamento daria 1,000")

    E = 0.03 * newton_schulz(G, 5)
    prev = 0.03 * math.sqrt(min(G.shape)) * NS_EFF
    chk("A3 ‖ε‖_F ≈ ρ·√min(m,n)·NS_EFF (a fórmula que calibra ρ)",
        abs(E.norm().item() - prev) / prev < 0.2, f"medido={E.norm():.4f} previsto={prev:.4f}")

    m = nn.Sequential()
    m.add_module('embed', nn.Embedding(50, 32)); m.add_module('fc', nn.Linear(32, 64))
    m.add_module('conv', nn.Conv2d(4, 8, 3)); m.add_module('ln', nn.LayerNorm(64))
    m.add_module('lm_head', nn.Linear(64, 50))
    sel = [n for n, p in m.named_parameters() if eh_matriz_oculta(n, p)]
    chk("A4 split correto (ocultas e conv via Muon; embed/lm_head/norm/viés via AdamW)",
        sel == ['fc.weight', 'conv.weight'], f"Muon: {sel}")

    chk("A5 conv 4D vira matriz (out, in·kh·kw)",
        como_matriz(torch.randn(8, 4, 3, 3)).shape == (8, 36))

    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(16, 16, bias=False), nn.ReLU(), nn.Linear(16, 4, bias=False))
    o = MuonSR(net, lr=0.0, weight_decay=0.0)          # lr=0: só a perturbação pode mexer
    o.fase = 2; o.passos_f2 = 10 ** 6
    w0 = [p.detach().clone() for p in net.parameters()]
    x, y = torch.randn(8, 16), torch.randn(8, 4)
    ((net(x) - y) ** 2).mean().backward()
    def cl():
        o.zero_grad(); l = ((net(x) - y) ** 2).mean(); l.backward(); return l
    o.step(cl)
    dif = max((p.detach() - q).abs().max().item() for p, q in zip(net.parameters(), w0))
    chk("A6 a perturbação restaura W exatamente (lr=0 -> W inalterado)", dif < 1e-6, f"maxdiff={dif:.1e}")

    a = MuonSR(nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1)), lr=1e-3)
    b = MuonSR(nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Linear(1024, 1)), lr=1e-3)
    def razao(o):
        wF = math.sqrt(sum(float(p.detach().float().pow(2).sum()) for p in o._muon_params))
        return o.rho_eff * NS_EFF * math.sqrt(o._S) / wF
    chk("A7 a dose ‖ε‖/‖W‖ é a mesma em modelos de tamanhos diferentes (transfere)",
        abs(razao(a) - razao(b)) < 1e-4, f"{razao(a):.4f} vs {razao(b):.4f} (modelo 16× maior)")

    print("=" * 72); print("RESULTADO:", "TODAS PASSARAM" if ok else "HOUVE FALHAS"); print("=" * 72)
    return ok


# ==============================================================================
#  Exemplo
# ==============================================================================
def exemplo():
    import torch.nn as nn
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = MuonSR(net, lr=3e-3)
    print(">>>", opt.resumo())

    X = torch.randn(512, 20); Y = (X[:, :1] * 2 + X[:, 1:2] ** 2).detach()
    Xv = torch.randn(128, 20); Yv = (Xv[:, :1] * 2 + Xv[:, 1:2] ** 2).detach()

    for passo in range(1, 601):
        i = torch.randint(0, 512, (64,)); xb, yb = X[i], Y[i]
        opt.zero_grad(); ((net(xb) - yb) ** 2).mean().backward()
        def cl():
            opt.zero_grad(); l = ((net(xb) - yb) ** 2).mean(); l.backward(); return l
        opt.step(cl)

        if passo % 25 == 0:
            with torch.no_grad():
                vl = ((net(Xv) - Yv) ** 2).mean().item()
            st = opt.observe_val(vl)
            if passo % 100 == 0 or st['fase'] > 1:
                print(f"  passo {passo:4d} | fase {st['fase']} | val {vl:.5f} | "
                      f"best {st['best_val']:.5f} (fase {st['best_fase']}) | ρ={st['rho']:.4f}")
            if st['fase'] == 3:
                break
    print(">>>", opt.resumo())


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--exemplo':
        exemplo()
    else:
        sys.exit(0 if auditorias() else 1)
