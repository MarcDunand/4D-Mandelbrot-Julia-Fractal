import JuliaMandel


with open("mandelCmds.txt", "r") as file:
    lines = file.readlines()  # Reads all lines into a list

# Stripping newline characters
lines = [line.strip() for line in lines]
initVals = lines.pop(0)

commands = [line.split() for line in lines]  # Split each line into a list
initVals = initVals.split()

# Convert to the correct types
commands = [{'s':int(cmd[0]), 
             'e':int(cmd[1]), 
             'p':cmd[2], 
             'v':float(cmd[3])} for cmd in commands]

initVals = {'parameterization': initVals[0],
            'Yres': int(initVals[1]),
            'Xres': int(initVals[2]),
            'Hres': int(initVals[3]),

            'precision': int(initVals[4]),
            'time': float(initVals[5]),
            'Ymin': float(initVals[6]),
            'Ymax': float(initVals[7]),
            'Xmin': float(initVals[8]),
            'Xmax': float(initVals[9]),
            'Hmin': float(initVals[10]),
            'Hmax': float(initVals[11]),
            'XYrot': float(initVals[12]),
            'XTrot': float(initVals[13]),
            'XHrot': float(initVals[14]),
            'YTrot': float(initVals[15]),
            'YHrot': float(initVals[16]),
            'HTrot': float(initVals[17])}



maxT = 0
for cmd in commands:
    if cmd['e'] > maxT:
        maxT = cmd['e']

params = [[0]+[0.0]*13]*maxT

JuliaMandel.generateFromCode(initVals, params)