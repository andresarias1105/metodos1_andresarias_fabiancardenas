def gr_a_r(grados:float)->float:
    return grados*3.1416/180

grados = float(input('ingrese valor a convertir'))

radianes =round( gr_a_r(grados),2)

print(grados,' grados son ',radianes, ' radianes')