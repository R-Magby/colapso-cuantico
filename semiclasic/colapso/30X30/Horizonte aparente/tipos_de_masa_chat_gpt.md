¡Claro! En relatividad general, existen varias definiciones de **masa**, cada una adaptada a contextos específicos. Algunas son globales, como la **masa ADM** o la **masa de Bondi**, mientras que otras, como la **masa de Choptuik**, están relacionadas con fenómenos particulares, como el colapso gravitacional crítico. Aquí te doy una descripción de las más relevantes:

---

### 1. **Masa ADM (Arnowitt-Deser-Misner)**:
Esta masa describe la energía total de un espacio-tiempo asintóticamente plano y se calcula en el infinito espacial. Es adecuada para sistemas aislados y se define usando las propiedades de la métrica en el límite lejano (\(r $\to$ $\infty$\)):

$M_\text{ADM} = \frac{1}{16\pi} \int_{S^2_\infty} \left( \partial^j h_{ij} - \partial_i h^j_j \right) dS^i,$

donde $\(h_{ij} = \gamma_{ij} - \delta_{ij}\)$ representa las desviaciones de la métrica respecto a la métrica plana. Es útil en simulaciones con el formalismo ADM.

---

### 2. **Masa de Bondi**:
La masa de Bondi se utiliza para describir la energía de un espacio-tiempo en el infinito **futuro nulo** ($\(\mathscr{I}^+\)$), lo que la hace relevante para sistemas radiantes (por ejemplo, emisiones de ondas gravitacionales). Se define como:

$M_\text{Bondi} = \frac{1}{4\pi} \int_{S^2} \Psi_2 \, d\Omega$

donde $ \(\Psi_2\)$ es un componente del escalar de Weyl en el formalismo de Newman-Penrose, que describe las curvaturas gravitacionales lejanas.

---

### 3. **Masa de Komar**:
La masa de Komar es útil para sistemas estáticos o estacionarios. Está definida a partir de la simetría temporal del espacio-tiempo, usando el vector de Killing asociado a esta simetría ($\(\xi^\mu\)$):

$
M_\text{Komar} = \frac{1}{4\pi} \int_\Sigma R_{\mu\nu} n^\mu \xi^\nu \, \sqrt{\gamma} \, d^3x,
$

o, alternativamente, como una integral sobre el infinito espacial:

$
M_\text{Komar} = \frac{1}{8\pi} \int_{S^2_\infty} \nabla^\mu \xi^\nu dS_{\mu\nu}.
$

Esta definición es útil para agujeros negros estáticos o sistemas aislados estacionarios.

---

### 4. **Masa de Choptuik (Crítica del colapso gravitacional)**:
La masa de Choptuik surge en el contexto del **colapso gravitacional crítico** de un campo escalar. En este fenómeno, al ajustar las condiciones iniciales cerca de un umbral crítico, el sistema puede colapsar y formar un agujero negro. La masa del agujero negro cerca del umbral se comporta de manera universal según:

$
M_\text{BH} \propto (p - p_c)^\gamma,
$

donde:
- $\(p\)$ es un parámetro que describe las condiciones iniciales del sistema.
- $\(p_c\)$ es el valor crítico donde ocurre el colapso.
-$ \(\gamma\)$ es un exponente crítico universal (aproximadamente $\(0.37\)$ para un campo escalar masivo).

Este resultado muestra que, cerca del umbral crítico, la masa del agujero negro es extremadamente sensible a las condiciones iniciales. Es una propiedad emergente del colapso crítico y revela un comportamiento fractal en las soluciones.

---

### 5. **Masa de Hawking (Quasi-local)**:
La masa de Hawking se define en términos de una superficie cerrada $\(S\)$ (generalmente, una 2-esfera) en el espacio-tiempo. Es una masa **quasi-local**, que mide la energía contenida dentro de $\(S\)$, y se calcula como:

$
M_\text{Hawking} = \sqrt{\frac{A}{16\pi}} \left( 1 - \frac{1}{16\pi} \int_S \Theta_+ \Theta_- \, dA \right),
$

donde:
- $\(A\)$ es el área de la superficie $\(S\)$.
- $\(\Theta_+\)$ y $\(\Theta_-\)$ son las expansiones asociadas a los rayos de luz salientes e ingresantes.

Es útil en escenarios como la evolución de horizontes en agujeros negros o regiones altamente dinámicas del espacio-tiempo.

---

### 6. **Masa de Misner-Sharp**:
La masa de Misner-Sharp es relevante en sistemas esféricamente simétricos. Es una medida local de la energía dentro de un radio $ \(r\)$ en el espacio-tiempo y se relaciona con la curvatura:

$
M_\text{MS}(r) = \frac{r}{2} \left( 1 - g^{\mu\nu} \partial_\mu r \partial_\nu r \right).
$

Se utiliza frecuentemente en estudios de colapsos gravitacionales y métricas dinámicas como la de Lemaître-Tolman-Bondi.

---

### 7. **Masa irreducible (agujeros negros)**:
En agujeros negros, la **masa irreducible** está asociada al área del horizonte de eventos y no puede disminuir en procesos clásicos debido al teorema de área de Hawking. Se define como:

$
M_\text{ir} = \sqrt{\frac{A}{16\pi}},
$

donde $\(A\)$ es el área del horizonte. Este concepto es útil en la termodinámica de agujeros negros.

---

### Conexión con tu simulación:
Si estás estudiando un colapso gravitacional, la **masa de Choptuik** puede ser especialmente relevante si investigas el umbral crítico. Si buscas una masa total en un espacio curvado, la **masa ADM** o la **masa de Misner-Sharp** podrían ser más útiles dependiendo de si tu sistema es asintóticamente plano o sféricamente simétrico.

¿Quieres que profundicemos en alguna de estas masas o en cómo implementarlas en tu simulación?
