import JuliaMandel
from itertools import islice

record = True


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

maxT = 0
for cmd in commands:
    if cmd['e'] > maxT:
        maxT = cmd['e']


initVals = {'parameterization': initVals[0],
            'Yres': int(initVals[1]),
            'Xres': int(initVals[2]),
            
            'Hres': int(initVals[3]),
            'prec': int(initVals[4]),
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


params = [dict(islice(initVals.items(), 3, None))]*maxT

commands = sorted(commands, key=lambda x: x["s"])

paramsIdx = 1
cmdsIdx = 0
curParams = {"cur" : params[0], "delta" : {key : 0.0 for key in params[0]}, "frames" : {key : 0 for key in params[0]}}

while paramsIdx < len(params):
    while cmdsIdx < len(commands) and paramsIdx == commands[cmdsIdx]["s"]:
        cmd = commands[cmdsIdx]

        tDiff = cmd["e"] - cmd["s"]
        vDiff = cmd["v"] - curParams["cur"][cmd["p"]]
        slope = vDiff/tDiff

        curParams["delta"][cmd["p"]] = slope
        curParams["frames"][cmd["p"]] = tDiff

        cmdsIdx += 1

    curParams["cur"] = {k: curParams["cur"][k] + curParams["delta"][k] for k in curParams["cur"]}
    curParams["frames"] = {k: v - 1 if v > 0 else v for k, v in curParams["frames"].items()}

    for k, v in curParams["frames"].items():
        if v == 0:
            curParams["delta"][k] = 0.0


    
    params[paramsIdx] = curParams["cur"]
    paramsIdx+=1





JuliaMandel.generateFromCode(initVals, params, record)