ElasticShape: revisión matemática independiente — 2026-09-07

El ensanchado lineal y la corrección de atención son algebraicamente correctos. La preservación de la red completa y de su trayectoria de entrenamiento es aproximada. La revisión reproduce un defecto en la normalización del ruido y precisa otras limitaciones: LayerNorm, transporte de Adam, edad de las coordenadas nuevas, weight decay y medición de deriva.

Alcance: código actual de `src/mztrain/shape_ops.py` y `src/mztrain/elastic_shape.py`, sus tests y SPEC/RFC. Comprobaciones nuevas en CPU, PyTorch 2.11.0+cu128; float64 para identidades/derivadas y fp32 para conversión SVD y bloque identidad. No se repitieron los entrenamientos T8–T14. Los contraejemplos prueban límites de las garantías; no prueban que las corridas publicadas sufran esos problemas. No se modificó la implementación.

Reproducción desde `D:/mztrain`:

```powershell
& .\.venv\Scripts\python.exe .tmp/elasticshape-math-20260907/audit_math.py
```

El script terminó con código 0 y todas sus aserciones pasaron. Algunas aserciones verifican contraejemplos: pasarlas confirma la limitación descrita, no que el comportamiento sea deseable.

1. Ensanchado y conversión: identidades correctas

Sea W = U D V, con D = diag(S), U de forma m×r y V de forma r×n. Si existen gates, sustituir S por S·wake_gate; el código los conserva.

Sean P_in y P_out matrices de inserción con columnas ortonormales: P_in^T P_in = I y P_out^T P_out = I. Los mapas de índices del código implementan estas inserciones. Con ruido cero:

$$U'=P_{out}U,\qquad V'=VP_{in}^{T},\qquad b'=P_{out}b.$$

Para x' = P_in x + z, donde P_in^T z = 0:

$$U'DV'x'+b'=P_{out}(UDVx+b).$$

Se conserva la salida vieja incluso si las entradas nuevas contienen valores arbitrarios. Es una identidad sobre números reales; los kernels pueden cambiar el orden de sumas en punto flotante. La promesa de igualdad bit a bit de la SPEC es excesiva, aunque el propio código y otras partes de la SPEC ya reconocen la tolerancia numérica.

`dense_to_factorized` hace SVD completa: W = QΣR^T, luego U=Q, S=diag(Σ), V=R^T. Correcto salvo redondeo. El código fuerza la SVD a fp32 mediante W.float(): un modelo float64 no obtiene por ello precisión float64 de reconstrucción. En parámetros bf16 también hay redondeo al guardar factores; el banco bf16 publicado usa parámetros fp32 y autocast.

Ensanchar conserva r, por lo que rank(W') ≤ r. Crecen los espacios de entrada y salida y las posibles direcciones, pero no desaparece el límite de rango de cada mapa lineal. El rango efectivo de una matriz nunca supera min(m,n); usar más columnas internas que ese mínimo sí es matemáticamente posible como parametrización redundante. Es el constructor de MZTrain el que lo prohíbe. El RFC confunde estas dos afirmaciones cuando llama «inexpresable» a r>min(m,n).

Anclas: [ensanchado](D:/mztrain/src/mztrain/shape_ops.py:54), [SVD](D:/mztrain/src/mztrain/shape_ops.py:190).

2. Atención: corrección exacta bajo las hipótesis declaradas

Por cabeza, los scores son QK^T/√h. Al crecer h→h', rellenar con ceros sin compensación cambiaría la temperatura del softmax. Con c=√(h'/h), Q'=[cQ,0] y K'=[K,0]:

$$\frac{Q'K'^T}{\sqrt{h'}}=\frac{cQK^T}{\sqrt{h'}}=\frac{QK^T}{\sqrt h}.$$

Los pesos de atención son iguales. V'=[V,0] conserva las salidas viejas, y el mapa intercalado de la proyección de salida vuelve a colocarlas correctamente. Se debe escalar el bias de Q si existe. Los mapas y el escalado implementados cumplen esto.

Con ruido en Q y K nuevos aparece un término adicional Q_new K_new^T/√h': la igualdad deja de ser exacta; con factores acotados ese término es de segundo orden en la amplitud del ruido. La deriva de LayerNorm puede ser de primer orden y dominarlo.

Prueba aislada, sin LayerNorm ni ruido: error relativo de salida 1.49e-16. Por autograd, el gradiente respecto a las filas U_q migradas cambia por 1/c; error máximo frente a esa fórmula 2.66e-15. Esto valida la corrección local, sin extenderla a toda la red transformada.

La definición usada coincide con [SDPA de PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html). Ancla: [corrección Q](D:/mztrain/src/mztrain/elastic_shape.py:194).

3. LayerNorm: fórmula exacta del error residual

Para x de dimensión d, sean μ su media y σ² su varianza con divisor d. Es la convención de [LayerNorm de PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html), basada en [Ba et al.](https://arxiv.org/abs/1607.06450). Definamos ρ=d/d', con d'>d. Al rellenar x con ceros:

$$\mu'=\rho\mu,\qquad \sigma'^2=\rho\sigma^2+\rho(1-\rho)\mu^2.$$

El código copia β, mantiene ε y usa γ'=√ρ γ en las coordenadas viejas. Sustituyendo y simplificando, su salida allí es exactamente:

$$y_i'=\gamma_i\frac{x_i-\rho\mu}{\sqrt{\sigma^2+(1-\rho)\mu^2+\varepsilon/\rho}}+\beta_i.$$

La salida original es:

$$y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta_i.$$

Hay tres diferencias: desplazamiento (1−ρ)μ en el numerador, varianza adicional (1−ρ)μ² y ε/ρ en lugar de ε. Si |μ|≪σ y ε≪σ², el término dominante de y'−y es γ(1−ρ)μ/σ. La compensación es razonable en ese régimen. Incluso con μ=0, mantener ε impide la igualdad exacta en general.

Contraejemplo con γ=1, β=0, ε=1e-5 y d=4→8: x=(1,1,1,1). LayerNorm original devuelve cero. La compensada devuelve aproximadamente 0.707093 en cada coordenada vieja. Con x=(1,2,3,4), el error relativo en esas coordenadas es 75.75%; con x+100 es 121.21%. La LayerNorm original es invariante a sumar una constante a x; después del relleno con ceros esa invariancia se pierde.

La fórmula coincide con la ejecución a 1.11e-16. Por tanto, las bandas pequeñas de deriva publicadas son observaciones de modelos y datos concretos; no una cota universal del método. El comportamiento de μ/σ también importa: aumentar d por sí solo no garantiza reducir la deriva si las componentes están correlacionadas.

No hay una elección universal de γ y β fijos que restaure la LayerNorm original para todas las x manteniendo este relleno con ceros. Eso no demuestra que toda forma de crecimiento con normalización exacta sea imposible: cambiar la normalización o la construcción del embedding del estado abre otras posibilidades.

Ancla: [compensación](D:/mztrain/src/mztrain/shape_ops.py:134).

4. Ruido: falta una cota independiente de la representación de los factores

El código genera filas nuevas ΔU con desviación ξ·std(U), donde ξ=noise_scale. Para una entrada fija, h=DVx, y k filas nuevas independientes:

$$\mathbb E\|\Delta U h\|_2^2=k\xi^2\operatorname{std}(U)^2\|h\|_2^2.$$

Una cota determinista condicionada a los factores y al ruido realizado sería:

$$\|\Delta UDVx\|_2\leq\|\Delta U\|_2\|D\|_2\|V\|_2\|x\|_2.$$

Ninguna de las dos expresiones depende solo de ξ. Además, los factores no son únicos: multiplicar una columna U[:,j] por a y dividir S[j] por a deja W exactamente igual. Sin embargo, cambia std(U), usada globalmente para todas las columnas de las filas nuevas. En otro componente k≠j, S[k] no disminuye y el ruido funcional puede crecer proporcionalmente a a.

Contraejemplo reproducido con rango 2, ξ=1e-3 y el mismo generador de ruido:

| Cantidad | Representación inicial | Misma W con una columna reescalada 1e6× |
|---|---:|---:|
| Norma de salidas nuevas / norma de salidas viejas | 0.0011587 | 368.3888 |
| Diferencia relativa entre las funciones antes de crecer | referencia | 1.42e-16 |

La salida vieja de la capa se conserva en ambas; lo que explota es la salida nueva. Esta prueba refuta una garantía uniforme «noise_scale pequeño implica perturbación funcional pequeña». No demuestra que una SVD recién creada esté mal condicionada: recién convertida empieza con factores ortogonales; el problema puede aparecer después de entrenar y antes de otro growth. U y V no se mantienen ortogonales durante el entrenamiento ordinario, y S deja de ser necesariamente el espectro singular real de W.

Para endurecer matemáticamente el mecanismo habría que controlar el ruido por efecto funcional, calibrarlo con activaciones o normalizar la representación de factores con transporte de estado adecuado. Un ruido gaussiano, además, ofrece garantías probabilísticas si no se recorta, no una cota absoluta por amplitud.

Anclas: [std(U)](D:/mztrain/src/mztrain/shape_ops.py:84), [afirmación del RFC](D:/mztrain/docs/RFC-ELASTICSHAPE-1.md:97).

5. Adam: tres distinciones necesarias

Las ecuaciones de momentos y corrección de sesgo parten de [Kingma y Ba](https://arxiv.org/abs/1412.6980). Las conclusiones siguientes se derivaron para el cambio de coordenadas de ElasticShape.

Si θ'=cθ y se conserva localmente la misma función con L'(θ')=L(θ'/c), entonces g'=g/c. Transportar momentos como m'=m/c y v'=v/c² es correcto para esa transformación diagonal fija de los gradientes históricos. Esto justifica `_rescale_q_state` en la atención aislada.

Pero para c>0, omitiendo ε y weight decay:

$$\Delta\theta'=-\eta\frac{\hat m/c}{\sqrt{\hat v/c^2}}=-\eta\frac{\hat m}{\sqrt{\hat v}}=\Delta\theta.$$

Para conservar la trayectoria bajo θ'=cθ se necesitaría Δθ'=cΔθ. Adam con el mismo LR no cumple esa propiedad. La prueba con c=√2 da iguales desplazamientos de parámetros y un desplazamiento funcional de 0.707107 veces el original. Es una limitación de equivariancia del optimizador, no un error de signo en la corrección Q. En ese caso aislado, LR'=c·LR y ε'=ε/c restablecerían el término adaptativo transformado; con weight decay habría que ajustar también su producto con LR. No es una solución global para una red cuyo LayerNorm también cambia.

Además, la compensación de LayerNorm transforma γ'=√ρ γ pero el código copia sus momentos sin reescalarlos. En el caso controlado μ=0, ε=0, pérdida que lee solo coordenadas viejas y activaciones fijas, se obtiene exactamente g_γ'=g_γ/√ρ. El transporte local análogo sería m_γ'=m_γ/√ρ y v_γ'=v_γ/ρ. Con ρ=1/2, autograd confirmó factor 1.414214. Esto identifica una inconsistencia local pendiente en la migración; con deriva real no basta ese factor para convertirla en exacta.

Por último, las coordenadas nuevas reciben m=v=0 y heredan la edad t del tensor. Para n gradientes constantes iguales a g≠0 después del crecimiento, sin weight decay y despreciando ε:

$$\frac{|\Delta\theta_{t+n}|}{\eta}=\frac{1-\beta_1^n}{1-\beta_1^{t+n}}\sqrt{\frac{1-\beta_2^{t+n}}{1-\beta_2^n}}.$$

Con β1=.9, β2=.999 y t muy grande: primer update 3.1623·η; update número 12, 6.5685·η. Con t=1000: 2.5153·η y 5.2412·η respectivamente. AdamW ejecutado con gradiente unitario reproduce estas cifras. Los 3.16 del código describen correctamente el límite del primer paso, pero no acotan los pasos siguientes ni demuestran estabilidad del entrenamiento. El warmup lineal de 200 pasos, suelo .1, reduce el transitorio; en este ejemplo a edad grande aún llega a 2.3377 veces el LR base durante la rampa.

Anclas: [estado Q](D:/mztrain/src/mztrain/elastic_shape.py:282), [momentos de LayerNorm](D:/mztrain/src/mztrain/elastic_shape.py:191), [padding Adam](D:/mztrain/src/mztrain/shape_ops.py:253).

La conversión densa→factorizada tiene una sutileza adicional. Reiniciar Adam es una decisión defendible, pero no es correcto decir que transportar cualquier información carece de sentido matemático. Para G=∂L/∂W y D=diag(S):

$$g_U=GV^TD,\qquad g_V=DU^TG,\qquad g_S=\operatorname{diag}(U^TGV^T).$$

Un primer momento denso puede proyectarse por estas transformaciones en los factores actuales como elección de transporte. Para el segundo momento exacto se requieren covarianzas de entradas de G que Adam diagonal no almacena. No hay una migración general exacta de toda la dinámica a partir de sus dos buffers diagonales; aproximaciones son posibles. Los embeddings y LayerNorm que no cambian durante la conversión podrían conservar su historial por identidad, aunque la política actual reinicia también ese estado.

6. Weight decay: el mismo número cambia la contracción funcional

[AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) aplica contracción desacoplada a cada parámetro. Aislando ese término, con a=1−ηλ:

$$W_{dense}^{+}=aW,\qquad W_{fact}^{+}=(aU)(aD)(aV)=a^3W.$$

A primer orden, a³≈1−3ηλ. Conservar λ al factorizar triplica aproximadamente la tasa de contracción de la matriz efectiva. Ejecución con η=.1 y λ=.2: denso 0.980000, factorizado 0.941192, como predice 0.98³.

Para igualar solamente este componente con LR constante, usar en cada factor λ_f=[1−(1−ηλ_d)^(1/3)]/η, aproximadamente λ_d/3. No iguala el efecto de los gradientes ni convierte la penalización de factores en una penalización equivalente sobre W. Esta observación no invalida comparar dos endpoints factorizados iguales, pero introduce otra diferencia entre la fase densa y la factorizada que debe separarse si se atribuye el mecanismo causal.

Ancla: [reconstrucción de AdamW](D:/mztrain/src/mztrain/elastic_shape.py:359).

7. Profundización: identidad correcta, activación gradual del gradiente

En un bloque pre-LN, anular U y bias en las proyecciones de salida de ambas ramas residuales da x+0+0=x, suponiendo activaciones intermedias finitas. El hecho de que las otras matrices nazcan aleatorias no altera la identidad.

Para una rama y=UDVx con U=0, g_U puede ser no nulo mientras g_S=g_V=0 y el gradiente hacia los módulos anteriores de esa rama es cero. Si U y V fueran ambos cero, también g_U sería cero y el camino quedaría bloqueado. La inicialización elegida deja S,V vivos para que U despierte primero.

La ejecución confirmó igualdad exacta de logits antes de entrenar. En el primer backward del bloque nuevo solo `proj.U` y `fc2.U` tuvieron gradiente no nulo; el resto tuvo gradiente cero. «El bloque despierta por gradiente» es correcto. «Todos sus parámetros reciben señal desde el primer paso» sería falso. El mismo razonamiento de producto se aplica a dimensiones ensanchadas: tener ruido en un lado puede habilitar primero el aprendizaje del lector situado al otro lado.

Ancla: [bloques identidad](D:/mztrain/src/mztrain/shape_ops.py:212).

8. Qué prueban la métrica de deriva y el ahorro de parámetros

La sonda usa δ=||z'−z||/||z||. Es útil para comparar logits con escala controlada; no garantiza una variación pequeña de pérdida. El softmax es invariante a sumar una constante a todos los logits de un token y esta razón no lo es. Ejemplo comprobado: z=(1000,1001), z'=(1001,1000). Deriva relativa 0.09995%, pero la entropía cruzada para clase 0 cambia exactamente 1 nat. Al aumentar la constante compartida, la razón tiende a cero sin reducir el cambio de pérdida.

Por el gradiente de entropía cruzada p−e_y, cuyo valor absoluto sumado es ≤2, una cota por token válida es |ΔCE|≤2||Δz||_∞. Esta cota depende de una diferencia absoluta, no solo de la deriva relativa. Para una garantía práctica de continuidad conviene medir Δloss y/o KL por token, junto con un criterio explícito de aceptación. `apply_event` actualmente informa la deriva y no aplica ese criterio.

Una capa densa m×n sin bias tiene mn parámetros; la factorizada tiene r(m+n+1). Hay ahorro si y solo si r<mn/(m+n+1). Convertir una capa cuadrada completa a r=d produce 2d²+d parámetros frente a d²: la conversión exacta sola no comprime.

En un bloque GPT del código, las cuatro matrices densas suman 12d². Un bloque de ancho D y rango común r suma 16Dr+4r. Si D=kd y r=d, el cociente contra el bloque denso de ancho D es 4/(3k)+1/(3k²d), aproximadamente 2/3 para k=2. Se excluyen aquí embeddings, LayerNorm, estados, activaciones y costes de atención. La proyección cuadrada individual a D=2d tiene 4d²+d parámetros: no ahorra frente a su equivalente denso, aunque el bloque en conjunto sí.

El ahorro de reloj exige que el aprendizaje barato compense los pasos restantes después de crecer. Con a pasos pequeños, b pasos grandes, costes medios C_s,C_g y cirugía C_c:

$$T_M=aC_s+bC_g+C_c.$$

Para una referencia de K pasos grandes, ganar exige T_M<KC_g. El álgebra de ensanchado no demuestra cuántos b harán falta para alcanzar la calidad objetivo. Los ratios de T8–T12 sustentan esa parte empíricamente dentro de su protocolo.

Anclas: [sonda](D:/mztrain/src/mztrain/elastic_shape.py:299), [arquitectura del bloque](D:/mztrain/src/mztrain/elastic_shape.py:46).

Prioridades que se desprenden de las pruebas: controlar el ruido por su efecto funcional; explicitar la deriva de LayerNorm con la fórmula completa; revisar el transporte de γ y el transitorio de Adam; separar el efecto del weight decay en los experimentos; complementar la sonda de logits con pérdida o KL. Las identidades lineales y de atención proporcionan una base válida para ese trabajo.
