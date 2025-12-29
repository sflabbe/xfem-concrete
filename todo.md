
## 1) Insights sobre el problema actual (más allá del s=0 singular)

### A. El síntoma “K_concrete diag ~ 6 N/m” es casi seguro un bug de escala

Orden de magnitud esperable (2D plane stress/strain Q4) para un elemento típico:

[
K \sim \frac{E,t,A}{L^2};;\approx;;\frac{E,t}{L}
]

Si (E\approx 30\text{ GPa}=3\cdot 10^{10},\mathrm{N/m^2}), (t\sim 0.2,\mathrm{m}), (L\sim 0.1,\mathrm{m}),
[
K \sim \frac{3\cdot10^{10}\cdot0.2}{0.1}\approx 6\cdot 10^{10},\mathrm{N/m}
]
No **6**.

**Que te salga 6 N/m** sugiere (casi siempre) uno de estos:

* **Espesor (t)** está ~(10^{-10})–(10^{-11},\mathrm{m}) (olvido de thickness, o conversión mm→m al revés).
* **E en MPa** fue tratado como **Pa** (factor (10^{6})).
* **Geometría en mm** mezclada con (E) en Pa (o viceversa).
* Error en **Jacobian detJ** (mapeo Q4): detJ demasiado pequeño.
* B-matrix mal escalada (derivadas en (\xi,\eta) no convertidas a (x,y)).

👉 **Prioridad #1**: antes de tocar más bond-slip, arreglar esa escala. Mientras el “bulk” sea 10–12 órdenes más blando de lo que debería, el acoplamiento bond siempre te mata el Newton.

---

### B. Aun regularizado, (k_\text{bond}\sim 10^8) N/m puede ser “ok”… si el concreto está en 10¹⁰–10¹² N/m

Tu regularización bajó 6 órdenes, bien. Pero si el concreto está a 6 N/m, sigues con un gap de 7–10+ órdenes => **condicionamiento brutal** y Newton muere “antes de empezar”.

---

### C. Ojo con `steel_EA=0`: puedes introducir un mecanismo (modo casi rígido) al inicio

Si el acero no tiene rigidez axial y el bond al inicio es pequeño o “capado”, puedes crear un subespacio mal condicionado (o casi singular) en los DOFs del acero. En un pull-out, por ejemplo, necesitas:

* o **alguna rigidez axial mínima** del acero,
* o **condiciones de borde** que eliminen el modo rígido,
* o **condensación** de DOFs del acero (más abajo).

No es que `steel_EA=0` esté “mal”, pero te obliga a que el interfaz sea numéricamente saludable y que el sistema no tenga modos rígidos sueltos.

---

### D. Solución numérica estándar: **equilibración / scaling + condensación** (antes de “cambiar de solver”)

Dos fixes que suelen “salvar” estos problemas sin reescribir física:

1. **Diagonal scaling (equilibration)** del sistema lineal en cada Newton:
   [
   \tilde K = D^{-1/2} K D^{-1/2},\quad \tilde r = D^{-1/2} r
   ]
   donde (D=\mathrm{diag}(|K|)+\epsilon). Resolver (\tilde K,\tilde x=\tilde r) y desescalar (x=D^{-1/2}\tilde x).

2. **Static condensation del acero** (si el acero solo se conecta por bond):
   Particiona:
   [
   \begin{bmatrix}K_{cc} & K_{cs}\ K_{sc} & K_{ss}\end{bmatrix}
   \begin{bmatrix}u_c\ u_s\end{bmatrix}=
   \begin{bmatrix}r_c\ r_s\end{bmatrix}
   ]
   Condensa:
   [
   (K_{cc}-K_{cs}K_{ss}^{-1}K_{sc})u_c = r_c - K_{cs}K_{ss}^{-1}r_s
   ]
   Esto reduce el tamaño y muchas veces mejora conditioning (y evita DOFs “flotantes” del acero).

---

## Equation Pack (mínimo necesario para implementar todo sin tesis)

### 3.1. Elemento Q4 (bulk) — rigidez y residuo

Para cada elemento 2D (plane stress/strain), con espesor (t):

[
K_e = \int_{\Omega_e} B^T D_\text{tang} B; t; d\Omega
]
[
f^\text{int}*e = \int*{\Omega_e} B^T \sigma; t; d\Omega
]

En Gauss:
[
K_e \approx \sum_{gp} B_{gp}^T D_{gp} B_{gp}; t; \det J_{gp}; w_{gp}
]
[
f^\text{int}*e \approx \sum*{gp} B_{gp}^T \sigma_{gp}; t; \det J_{gp}; w_{gp}
]

Sanity scaling:
[
|K|\sim \frac{E,t}{L}
]

---

### 3.2. Bond-slip: kinemática, ley constitutiva, ensamblaje

**Slip (escalares)**
Para una barra con tangente unitaria (\mathbf{t}) (en global 2D):
[
s(x)=u_s(x) - \mathbf{t}^T \mathbf{u}_c(x)
]
donde (u_s) es desplazamiento axial del acero (1D) y (\mathbf{u}_c=[u_x,u_y]^T).

**Fuerza por unidad de longitud**
Model Code da (\tau(s)) (shear stress en interfaz). Fuerza lineal:
[
q(s)=\tau(s),p
]
con perímetro (p=\pi d_b) (aprox. barra circular).

**Energía disipada (interfaz)**
[
W_\text{bond} = \int q(s), ds = \int \tau(s),p; ds
]

**Forma débil / residuo**
Con funciones de forma 1D (N_i) sobre la barra (y proyección a concreto):
[
\delta s = \delta u_s - \mathbf{t}^T \delta \mathbf{u}_c
]
[
\delta W = \int \delta s; q(s); dx
]
De aquí:

* contribución al acero:
  [
  f_s = \int N_s^T, q(s); dx
  ]
* contribución al concreto (vectorial):
  [
  \mathbf{f}_c = -\int N_c^T, q(s),\mathbf{t}; dx
  ]

**Tangente consistente**
[
k_\tau(s)=\frac{d\tau}{ds},\quad k_q(s)=\frac{dq}{ds}=p,k_\tau(s)
]
[
K_{ss} = \int N_s^T, k_q, N_s; dx
]
[
K_{cc} = \int (N_c^T\mathbf{t}), k_q, (\mathbf{t}^T N_c); dx
]
[
K_{sc} = -\int N_s^T, k_q, (\mathbf{t}^T N_c); dx,;;;K_{cs}=K_{sc}^T
]

---

### 3.3. Model Code 2010 bond law + regularización en (s\to 0)

Forma típica ascendente (la que te causó singularidad):
[
\tau(s)=\tau_\max \left(\frac{s}{s_1}\right)^\alpha,\quad \alpha=0.4
]
[
\frac{d\tau}{ds}=\tau_\max,\alpha, s_1^{-\alpha}, s^{\alpha-1}
\Rightarrow s^{-0.6}\to\infty
]

**Regularización lineal** (lo que ya hiciste):
Sea (s_\text{reg}=\eta s_1) (ej. (\eta=0.01)).
Define:
[
k_0=\frac{\tau(s_\text{reg})}{s_\text{reg}}
]
[
\tau(s)=
\begin{cases}
k_0,s & 0\le s \le s_\text{reg}\
\tau_\max (s/s_1)^\alpha & s>s_\text{reg}
\end{cases}
]

**Mejor**: hacerlo (C^1) en (s_\text{reg}) (recomendado)
Usar rama potencia con offset:
[
\tau(s)=\tau_\max\left(\frac{s+s_0}{s_1}\right)^\alpha
]
Elegir (s_0) tal que (d\tau/ds) en 0 sea finito (y/o empatar con (k_0) en (s_\text{reg})).

**Tangent cap** (salvavidas numérico):
[
k_\tau^\text{used}=\min\left(\frac{d\tau}{ds}, k_{\tau,\max}\right)
]

---

### 3.4. Dowel action (modelo mínimo implementable)

En una intersección grieta–barra, define salto relativo en dirección normal/tangencial a la grieta.

Sea (\mathbf{n}) normal a la grieta, (\mathbf{t}_c) tangente de la grieta (no confundir con barra).
Salto de desplazamiento del concreto a ambos lados: (\llbracket \mathbf{u}_c \rrbracket).

Un modelo simple de “pasador”:
[
F_d = k_d, (\mathbf{n}^T \llbracket \mathbf{u}_c \rrbracket)
]
o si quieres penalizar también tangencial:
[
\mathbf{F}_d = k_n (\mathbf{n}^T \llbracket \mathbf{u}\rrbracket)\mathbf{n}

* k_t (\mathbf{t}_c^T \llbracket \mathbf{u}\rrbracket)\mathbf{t}_c
  ]
  y ensamblas como “spring” entre los DOFs enriquecidos/duplicados del crack (o entre pares de puntos si lo discretizas).

Energía:
[
W_d=\frac12 k_n (\Delta_n)^2 + \frac12 k_t (\Delta_t)^2
]

---

### 3.5. Arc-length (Riks) — constraint y linealización mínima

Sea el equilibrio:
[
\mathbf{R}(\mathbf{u},\lambda)=\mathbf{f}_\text{int}(\mathbf{u})-\lambda \mathbf{P}=0
]

Restricción de longitud de arco:
[
g(\Delta\mathbf{u},\Delta\lambda)=\Delta\mathbf{u}^T\Delta\mathbf{u}
+\psi^2(\Delta\lambda)^2= \Delta l^2
]
(o con (\Delta \mathbf{P}) si quieres escalar por carga, pero esto basta para implementar)

Newton extendido:
[
\begin{bmatrix}
K & -\mathbf{P}\
2\Delta\mathbf{u}^T & 2\psi^2\Delta\lambda
\end{bmatrix}
\begin{bmatrix}
\delta \mathbf{u}\ \delta\lambda
\end{bmatrix}
=============

-\begin{bmatrix}
\mathbf{R}\ g-\Delta l^2
\end{bmatrix}
]

---

### 3.6. Near-tip “no singular” (cohesive-friendly)

En vez de branch LEFM (\sim \sqrt{r}), usa una base no singular tipo:
[
F(r,\theta)= r\sin\frac{\theta}{2}
]
y su gradiente (para B-matrix enriquecida):
[
\nabla F = \frac{\partial F}{\partial r}\nabla r+\frac{\partial F}{\partial \theta}\nabla \theta
]
con:
[
\frac{\partial F}{\partial r}=\sin\frac{\theta}{2},\quad
\frac{\partial F}{\partial \theta}=\frac{r}{2}\cos\frac{\theta}{2}
]
y (r,\theta) computados respecto al tip en el sistema local del tip.

---

### 3.7. Junction enrichment (coalescencia) — forma mínima

La idea práctica: cuando una grieta llega a otra, el campo de salto ya no se representa con “un solo Heaviside” del crack A o B, sino con una combinación que permite continuidad correcta alrededor del nodo de unión.

Implementación mínima (sin “exotismos”):

* Define dos funciones de salto (H_1(x), H_2(x)) asociadas a cada rama.
* En zona de junction, enriquece con ambas:
  [
  \mathbf{u}(x)=\sum_i N_i \mathbf{u}*i + \sum*{i\in \mathcal{E}_1} N_i(H_1(x)-H_1(x_i))\mathbf{a}_i

- \sum_{i\in \mathcal{E}_2} N_i(H_2(x)-H_2(x_i))\mathbf{b}_i
  ]
  Eso ya te permite representar “dos saltos” que se cruzan/unifican sin colapsar el sistema.

---

### 3.8. Refuerzo mesh-independent (Heaviside-style) — dos rutas

**Ruta A (rápida, robusta): “nodos virtuales” + proyección**

* Discretiza el acero como 1D con nodos propios (no FE mesh).
* Proyecta (\mathbf{u}_c(x)) desde el elemento de concreto que contiene el punto.
* Ensambla bond con las ecuaciones de 3.2. (Esto suele ser suficiente para producción.)

**Ruta B (más “XFEM puro”): enriquecimiento**

* Introduce DOFs de acero como enriquecimiento sobre los elementos atravesados por la barra.
* Similar a crack Heaviside, pero con soporte alrededor del refuerzo.

---

## Qué haría yo mañana (si quiero que esto funcione ya)

1. **Sanity check de Ke_bulk** (unidades/espesor/Jacobiano).
2. **Equilibration + (opcional) condensación del acero**.
3. Solo después: volver al pull-out y recién ahí ajustar (s_\text{reg}), cap de tangente y/o cambiar solver.

Si me pegas el snippet donde defines **E, unidades geométricas y thickness**, te puedo decir con alta probabilidad cuál es el factor de escala que está roto (pero ya con lo de arriba, el agent puede encontrarlo solo).
