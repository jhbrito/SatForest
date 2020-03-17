import numpy as np

Black = [0, 0, 0]
White = [255, 255, 255]
NODATA = [0,0,0]

Artificializados = [255, 0, 0]
Agricolas = [255, 255, 0]
Floresta = [0, 255, 0]
Humidas = [255, 0, 255]
Agua = [0, 0, 255]

# Color coding for all agriculture and forest classes
A01 = [255, 255, 160] #'2.1.0.00.0': 'Culturas temporárias de sequeiro e de regadio',
A02 = [255, 220, 140] #'2.1.3.01.1': 'Arrozais',
A03 = [255, 200, 120] #'2.2.1.00.0': 'Vinhas',
A04 = [255, 180, 100] #'2.2.2.00.0': 'Pomares',
A05 = [255, 160, 80]  #'2.2.3.00.0': 'Olivais',
A06 = [255, 140, 40]  #'2.3.1.01.1': 'Pastagens permanentes',
A07 = [255, 120, 20]  #'2.4.1.00.0': 'Culturas temporárias e/ou pastagens associadas a culturas permanentes'
A08 = [255, 100, 0]   #'2.4.2.01.1': 'Sistemas culturais e parcelares complexos',
A09 = [255, 255, 210] #'2.4.3.01.1': 'Agricultura com espaços naturais e semi-naturais',
A10 = [255, 255, 180] #'2.4.4.00.1': 'SAF de sobreiro',
A11 = [255, 255, 150] #'2.4.4.00.2': 'SAF de azinheira',
A12 = [255, 255, 120] #'2.4.4.00.3': 'SAF de outros carvalhos',
A13 = [255, 255, 90]  #'2.4.4.00.4': 'SAF de pinheiro manso',
A14 = [255, 255, 60]  #'2.4.4.00.5': 'SAF de outras espécies',
A15 = [255, 255, 30]  #'2.4.4.00.6': 'SAF de sobreiro com azinheira',
A16 = [255, 255, 0]   #'2.4.4.00.7': 'SAF de outras misturas',
F01 = [0, 255, 0]     #'3.1.1.00.1': 'Florestas de sobreiro',
F02 = [0, 255, 20]    #'3.1.1.00.2': 'Florestas de azinheira',
F03 = [0, 255, 40]    #'3.1.1.00.3': 'Florestas de outros carvalhos',
F04 = [0, 255, 60]    #'3.1.1.00.4': 'Florestas de castanheiro',
F05 = [0, 255, 80]    #'3.1.1.00.5': 'Florestas de eucalipto',
F06 = [0, 255, 100]   #'3.1.1.00.6': 'Florestas de espécies invasoras',
F07 = [0, 255, 120]   #'3.1.1.00.7': 'Florestas de outras folhosas',
F08 = [0, 255, 140]   #'3.1.2.00.1': 'Florestas de pinheiro bravo',
F09 = [0, 255, 160]   #'3.1.2.00.2': 'Florestas de pinheiro manso',
F10 = [0, 255, 180]   #'3.1.2.00.3': 'Florestas de outras resinosas',
F11 = [0, 255, 200]   #'3.2.1.01.1': 'Vegetação herbácea natural',
F12 = [0, 255, 220]   #'3.2.2.00.0': 'Matos',
F13 = [0, 255, 240]   #'3.3.0.00.0': 'Espaços descobertos ou com pouca vegetação',

COLOR_DICT = np.array([Black, White])
COS_COLOR_DICT = np.array([Artificializados, Agricolas, Floresta, Agua])

# Semi-final agregation color coding
SEMI_COLOR_DICT = np.array([Artificializados, Agricolas, F01, A11, A16, F05, F08, F12, F10, Agua])