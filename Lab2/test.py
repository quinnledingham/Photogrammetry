import math

f = 152

x = 59.043
y = 72.392

h = 0.3
H = 3
K = ((2410 * H)/(H**2 - 6*H + 250)) - ((2410*h)/(h**2 - 6*h + 250)) * (h/H)
#K = 29.7088
K = K * 1E-6

r = math.sqrt(x**2 + y**2)

d_r = -K*(r + (r**3/f**2))
d_d = K * (r/f)
print(d_d)

alpha = math.atan(r/f)
print(alpha)

r_prime = f * math.tan(alpha + d_d)

print(r_prime)

print("Solutions:")
print((r_prime / r) * x)
print((r_prime / r) * y)


x_atm = -x * K * (1 + (r**2/f**2))
y_atm = -y * K * (1 + (r**2/f**2))

#print(x_atm)
#print(y_atm)
print(x + x_atm)
print(y + y_atm)
