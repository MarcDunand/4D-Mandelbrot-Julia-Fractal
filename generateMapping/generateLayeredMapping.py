import argparse

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
        

if __name__=="__main__":
    main()