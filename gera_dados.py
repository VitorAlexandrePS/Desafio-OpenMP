import random
random.seed(42)
with open("dados.txt","w") as f:
    for _ in range(1000):
        f.write(f"{random.gauss(170,10):.1f} {random.gauss(70,12):.1f}\n")
