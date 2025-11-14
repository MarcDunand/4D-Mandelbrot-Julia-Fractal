import argparse
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=256)
    args = ap.parse_args()

    side = args.side

    N = side**3
    bitlen = math.ceil(math.log2(N))

    layers = []
    for i in range(bitlen+1):
        layers.append([])

    #sorts all numbers up to N by the number of 1s in its binary representation, this creates the layers
    for i in range(N+1):  #generally 0-24
        num1s = bin(i).count("1")
        layers[num1s].append(i)

    mapping = []
    for n in range(0, 3*side):
        for x in range(0, side):
            # Compute allowable y range
            y_min = max(0, n - x - (side-1))
            y_max = min(side-1, n - x)

            if y_min > y_max:
                continue  # no valid y for this x

            for y in range(y_min, y_max + 1):
                z = n - x - y  # guaranteed 0 <= z < m
                mapping.append((x, y, z))


    print(mapping[0:100])
        

if __name__=="__main__":
    main()