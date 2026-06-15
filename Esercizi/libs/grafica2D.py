#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Primi passi nella grafica 2D di Python con il pacchetto matplotlib
"""


import numpy as np
import matplotlib.pyplot as plt
#  %matplotlib notebook

x = [0, 1, 2, 3] 
y = [0, 1, 4, 9]
plt.plot(x, y)
plt.show()


# f(x) = x ** 2   per  -5 <= x <= 5
x = np.linspace(-5,5, 100)
plt.plot(x, x**2)
plt.show()

"""
Symbol  Description    Symbol  Description
b       blue           T       T
g       green          s       square
r       red            d       diamond
c       cyan           v       triangle (down)
m       magenta        ^       triangle (up)
y       yellow         <       triangle (left)
k       black          >       triangle (right)
w       white          p       pentagram
.       point          h       hexagram
o       circle         -       solid
x       x-mark         :       dotted
+       plus           -.      dashdot
*       star           --      dashed
"""

x = np.linspace(-5,5, 100)
plt.plot(x, x**2, 'g--')
plt.show()

x = np.linspace(-5,5,20)
plt.plot(x, x**2, 'ko')
plt.plot(x, x**3, 'r*')
plt.show()

# specificare le diomensioni della figura (di default valori in pollici)
plt.figure(figsize = (10,6))

x = np.linspace(-5,5,20)
plt.plot(x, x**2, 'ko')
plt.plot(x, x**3, 'r*')
plt.title(f'Plot of Various Polynomials from {x[0]} to {x[-1]}')
plt.xlabel('X axis', fontsize = 18)
plt.ylabel('Y axis', fontsize = 18)
plt.show()

# Stili di visualizzazione disponibili
print(plt.style.available)

# 'Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid',
# 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot',
# 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind',
# 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid',
# 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook',
# 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster',
# 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white',
# 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'

plt.style.use('seaborn-v0_8-poster')

plt.figure(figsize = (10,6))

x = np.linspace(-5,5,20)
plt.plot(x, x**2, 'ko')
plt.plot(x, x**3, 'r*')
plt.title(f'Plot of Various Polynomials from {x[0]} to {x[-1]}')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# aggiungere una legenda
plt.figure(figsize = (10,6))

x = np.linspace(-5,5,20)
plt.plot(x, x**2, 'ko', label = 'quadratic')
plt.plot(x, x**3, 'r*', label = 'cubic')
plt.title(f'Plot of Various Polynomials from {x[0]} to {x[-1]}')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend(loc = 2)
plt.show()

# modificare l'aspetto: griglie, limiti degli assi, titolo, etichette degli assi...
plt.figure(figsize = (10,6))

x = np.linspace(-5,5,100)
plt.plot(x, x**2, 'ko', label = 'quadratic')
plt.plot(x, x**3, 'r*', label = 'cubic')
plt.title(f'Plot of Various Polynomials from {x[0]} to {x[-1]}')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend(loc = 2)
plt.xlim(-6.6)
plt.ylim(-10,10)
plt.grid()
plt.show()

# tabelle di plot (subplots)
x = np.arange(11)
y = x**2

plt.figure(figsize = (14, 8))

plt.subplot(2, 3, 1)
plt.plot(x,y)
plt.title('Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()

plt.subplot(2, 3, 2)
plt.scatter(x,y)
plt.title('Scatter')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()

plt.subplot(2, 3, 3)
plt.bar(x,y)
plt.title('Bar')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()

plt.subplot(2, 3, 4)
plt.loglog(x,y)
plt.title('Loglog')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(which='both')

plt.subplot(2, 3, 5)
plt.semilogx(x,y)
plt.title('Semilogx')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(which='both')

plt.subplot(2, 3, 6)
plt.semilogy(x,y)
plt.title('Semilogy')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()

plt.tight_layout()  # questo serve a non far sovrapporre le sottofigure una all'altra

plt.show()

# esportazione dei grafici su file esterni
plt.figure(figsize = (8,6))
plt.plot(x,y)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('image.pdf') # oppure PNG, oppure GIF, ...

# Altri tipi di visualizzazione 2D:
# errorbar
# polar
# stem
# hist
# boxplot
# pie
 
