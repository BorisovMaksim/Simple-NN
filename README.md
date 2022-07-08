# Simple-NN

# Теоретический материал

## Определение

<span style="font-family:Papyrus; font-size:2em;">
    Функция $ f:\mathbb{R}^m\rightarrow\mathbb{R}^n$ дифференцируема в точке $x_0$, если 
    $$f(x_0 + h) = f(x_0) + \color{#348FEA}{\left[D_{x_0} f \right]} (h) + \bar{\bar{o}} \left(\left| \left| h\right|\right|\right), $$
    где $\color{#348FEA}{\big[D_{x_0} f\big]}$ - дифференциал функции $f$
    
</span>

## Градиент сложной функции

<span style="font-family:Papyrus; font-size:2em;">
Формула производной сложной функции
$$\left[D_{x_0} (\color{#5002A7}{u} \circ \color{#4CB9C0}{v}) \right](h) = \color{#5002A7}{\left[D_{v(x_0)} u \right]} \left( \color{#4CB9C0}{\left[D_{x_0} v\right]} (h)\right)$$
Пусть $f(x) = g(h(x))$, тогда
    
$$\left[D_{x_0} f \right] (x-x_0) = \langle\nabla_{x_0} f, x-x_0\rangle.$$
    
С другой стороны,

$$\left[D_{h(x_0)} g \right] \left(\left[D_{x_0}h \right] (x-x_0)\right) = \langle\nabla_{h_{x_0}} g, \left[D_{x_0} h\right] (x-x_0)\rangle = \langle\left[D_{x_0} h\right]^* \nabla_{h(x_0)} g, x-x_0\rangle.$$
То есть $\color{#FFC100}{\nabla_{x_0} f} = \color{#348FEA}{\left[D_{x_0} h \right]}^{*} \color{#FFC100}{ \nabla_{h(x_0)} }g $ - применение сопряженного к $D_{x_0}h$ 
линейного отображения к вектору $\nabla_{h(x_0)}g$
</span>




## Градиенты для типичных слоёв
- $x$ — вектор, а $v(x)$ – поэлементное применение $v$
 $$f(x)=u(v(x))\Rightarrow \color{#348FEA}{\nabla_{x_0} f  = v'(x_0) \odot \left[\nabla_{v(x_0)} u\right]}$$
- Умножение на матрицу справа
 $$f(X) = g(XW) \Rightarrow \color{#348FEA}{\nabla_{X_0} f = \left[\nabla_{X_0W} (g) \right] \cdot W^T}$$
 - Умножение на матрицу слева
  $$f(W) = g(XW) \Rightarrow  \color{#348FEA}{\nabla_{X_0} f = X^T \cdot \left[\nabla_{XW_0} (g)\right]}$$







#  Архитектура сети

|   Input Layer |  First Hidden Layer   | Second Hidden Layer   |  Output Layer  |  
|---|:---:|---|---|
| $$A^{[0]} = X$$                   | $$Z^{[1]} = W^{[1] } A^{[0]} + b^{[1]}$$   |  $$Z^{[2]} = W^{[2] } A^{[1]} + b^{[2]}$$   |    $$Z^{[3]} = W^{[3] } A^{[2]} + b^{[3]}$$     |
|                                   | $$A^{[1]} = ReLU(Z^{[1]})$$                 |  $$A^{[2]} = ReLU(Z^{[2]})$$                 |    $$A^{[3]} = Sigmoid(Z^{[3]})$$                |   
|                                   | $$W^{[1]}.shape = (n_{h}^{[1]}, n_x) $$               |  $$W^{[2]}.shape = (n_{h}^{[2]}, n_{h}^{[1]}) $$               |   $$W^{[3]}.shape = (n_{y}, n_{h}^{[2]}) $$                  |
| $$A^{[0]}.shape = (n_x, m) $$   | $$ Z^{[1]}.shape = (n_{h}^{[1]}, m) $$            |  $$ Z^{[2]}.shape = (n_{h}^{[2]}, m) $$            |      $$ Z^{[3]}.shape = (n_{y}, m) $$            |



# Backpropagation

$$ \Large  \mathcal{L}(A^{[3]}, Y) = -\frac{1}{m} \sum_{i=1}^{m}y_i log(A^{[3]}_i) + (1 - y_i) log(1 - A^{[3]}_i) = -\frac{1}{m}( Y^T log(A^{[3]}) + (1 - Y)^T log(1 - A^{[3]})) $$

$$ \Large  A^{[3]}= Sigmoid(Z^{[3]})$$

 $$ \Large  Z^{[3]} = W^{[3] } A^{[2]} + b^{[3]}$$ 
 
  $$ \Large  A^{[2]} = ReLU(Z^{[2]})$$    
  
   $$ \Large  Z^{[2]} = W^{[2] } A^{[1]} + b^{[2]}$$   
   
   $$  \Large A^{[1]} = ReLU(Z^{[1]})$$ 
   
   $$  \Large  Z^{[1]} = W^{[1] } A^{[0]} + b^{[1]}$$ 
   
- $\large \frac{\partial \mathcal{L}}{\partial W^{[3]}} = \frac{1}{m} \left(A_{0}^{[3]} - Y \right) A_{0}^{[2]T}$
- $\large \frac{\partial \mathcal{L}}{\partial b^{[3]}} = \frac{1}{m} np.sum(A^{[3]}_{0} - Y , axis=1, keepdims=True)$ 
- $\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \frac{1}{m} np.heaviside(Z_{0}^{[2]}, 0)\odot W_{0}^{[3]T}(A_{0}^{[3]} - Y)A_{0}^{[1]T}$
- $\frac{\partial \mathcal{L}}{\partial b^{[2]}} = \frac{1}{m} np.sum(np.heaviside(Z_{0}^{[2]}, 0) \odot W_{0}^{[3]T}(A^{[3]}_{0} - Y), axis = 1, keepdims=True)$
- $\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{1}{m} np.heaviside(Z_{0}^{[1]}, 0) \odot W_{0}^{[2]T}  np.heaviside(Z_{0}^{[2]}, 0) \odot  W_{0}^{[3]T} (A_{0}^{[3]} - Y )  A_{0}^{[0]T}$ 
- $\frac{\partial \mathcal{L}}{\partial b^{[1]}} = \frac{1}{m} np.sum(np.heaviside(Z_{0}^{[1]}, 0) \odot W_{0}^{[2]T}  np.heaviside(Z_{0}^{[2]}, 0) \odot  W_{0}^{[3]T} (A_{0}^{[3]} - Y ), axis=1, keepdims=True)$ 
   

