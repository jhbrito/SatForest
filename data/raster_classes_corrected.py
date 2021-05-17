classes = [
  '1.1.1.00.0',
  '1.1.2.00.0',
  '1.2.1.00.0',
  '1.2.2.00.0',
  '1.2.3.00.0',
  '1.2.4.00.0',
  '1.3.1.00.0',
  '1.3.2.00.0',
  '1.3.3.00.0',
  '1.4.1.00.0',
  '1.4.2.01.1',
  '1.4.2.02.0',
  '1.4.2.03.0',
  '2.1.0.00.0',
  '2.1.3.01.1',
  '2.2.1.00.0',
  '2.2.2.00.0',
  '2.2.3.00.0',
  '2.3.1.01.1',
  '2.4.1.00.0',
  '2.4.2.01.1',
  '2.4.3.01.1',
  '2.4.4.00.1',
  '2.4.4.00.2',
  '2.4.4.00.3',
  '2.4.4.00.4',
  '2.4.4.00.5',
  '2.4.4.00.6',
  '2.4.4.00.7',
  '3.1.1.00.1',
  '3.1.1.00.2',
  '3.1.1.00.3',
  '3.1.1.00.4',
  '3.1.1.00.5',
  '3.1.1.00.6',
  '3.1.1.00.7',
  '3.1.2.00.1',
  '3.1.2.00.2',
  '3.1.2.00.3',
  '3.2.1.01.1',
  '3.2.2.00.0',
  '3.3.0.00.0',
  '4.0.0.00.0',
  '5.1.1.00.0',
  '5.1.2.00.0',
  '5.2.1.01.1',
  '5.2.2.01.1',
  '5.2.3.01.1',
]

class_lookup = {
  0: '1.1.1.00.0', # '1.1.1.00.0': 'Tecido urbano contínuo',
  1: '1.1.2.00.0', # '1.1.2.00.0': 'Tecido urbano descontínuo',
  2: '1.2.1.00.0', # '1.2.1.00.0': 'Indústria, comércio e equipamentos gerais',
  3: '1.2.2.00.0', # '1.2.2.00.0': 'Rede viária e espaços associados',
  4: '1.2.3.00.0', # '1.2.3.00.0': 'Áreas portuárias',
  5: '1.2.4.00.0', # '1.2.4.00.0': 'Aeroportos e aeródromos',
  6: '1.3.1.00.0', # '1.3.1.00.0': 'Áreas de extracção de inertes',
  7: '1.3.2.00.0', # '1.3.2.00.0': 'Áreas de deposição de resíduos',
  8: '1.3.3.00.0', # '1.3.3.00.0': 'Áreas em construção',
  9: '1.4.1.00.0', # '1.4.1.00.0': 'Espaços verdes urbanos',
  10: '1.4.2.01.1', # '1.4.2.01.0': 'Campos de golfe',
  11: '1.4.2.02.0', # '1.4.2.02.0': 'Outras Instalações desportivas e equipamentos de lazer',
  12: '1.4.2.03.0', # '1.4.2.03.0': 'Outros equipamentos culturais e outros e zonas históricas',
  13: '2.1.0.00.0', # '2.1.0.00.0': 'Culturas temporárias de sequeiro e de regadio',
  14: '2.1.3.01.1', # '2.1.3.01.1': 'Arrozais',
  15: '2.2.1.00.0', # '2.2.1.00.0': 'Vinhas',
  16: '2.2.2.00.0', # '2.2.2.00.0': 'Pomares',
  17: '2.2.3.00.0', # '2.2.3.00.0': 'Olivais',
  18: '2.3.1.01.1', # '2.3.1.01.1': 'Pastagens permanentes',
  19: '2.4.1.00.0', # '2.4.1.00.0': 'Culturas temporárias e/ou pastagens associadas a culturas permanentes'
  20: '2.4.2.01.1', # '2.4.2.01.1': 'Sistemas culturais e parcelares complexos',
  21: '2.4.3.01.1', # '2.4.3.01.1': 'Agricultura com espaços naturais e semi-naturais',
  22: '2.4.4.00.1', # '2.4.4.00.1': 'SAF de sobreiro',
  23: '2.4.4.00.2', # '2.4.4.00.2': 'SAF de azinheira',
  24: '2.4.4.00.3', # '2.4.4.00.3': 'SAF de outros carvalhos',
  25: '2.4.4.00.4', # '2.4.4.00.4': 'SAF de pinheiro manso',
  26: '2.4.4.00.5', # '2.4.4.00.5': 'SAF de outras espécies',
  27: '2.4.4.00.6', # '2.4.4.00.6': 'SAF de sobreiro com azinheira',
  28: '2.4.4.00.7', # '2.4.4.00.7': 'SAF de outras misturas',
  29: '3.1.1.00.1', # '3.1.1.00.1': 'Florestas de sobreiro',
  30: '3.1.1.00.2', # '3.1.1.00.2': 'Florestas de azinheira',
  31: '3.1.1.00.3', # '3.1.1.00.3': 'Florestas de outros carvalhos',
  32: '3.1.1.00.4', # '3.1.1.00.4': 'Florestas de castanheiro',
  33: '3.1.1.00.5', # '3.1.1.00.5': 'Florestas de eucalipto',
  34: '3.1.1.00.6', # '3.1.1.00.6': 'Florestas de espécies invasoras',
  35: '3.1.1.00.7', # '3.1.1.00.7': 'Florestas de outras folhosas',
  36: '3.1.2.00.1', # '3.1.2.00.1': 'Florestas de pinheiro bravo',
  37: '3.1.2.00.2', # '3.1.2.00.2': 'Florestas de pinheiro manso',
  38: '3.1.2.00.3', # '3.1.2.00.3': 'Florestas de outras resinosas',
  39: '3.2.1.01.1', # '3.2.1.01.1': 'Vegetação herbácea natural',
  40: '3.2.2.00.0', # '3.2.2.00.0': 'Matos',
  41: '3.3.0.00.0', # '3.3.0.00.0': 'Espaços descobertos ou com pouca vegetação',
  42: '4.0.0.00.0', # '4.0.0.00.0': 'Zonas húmidas',
  43: '5.1.1.00.0', # '5.1.1.00.0': 'Cursos de água naturais',
  44: '5.1.2.00.0', # '5.1.2.00.0': 'Planos de água',
  45: '5.2.1.01.1', # '5.2.1.01.1': 'Lagoas costeiras',
  46: '5.2.2.01.1', # '5.2.2.01.1': 'Desembocaduras fluviais',
  47: '5.2.3.01.1', # '5.2.3.01.1': 'Oceano',
  99: 'NODATA',
}

# Create QGis field calculator command string.
command = 'CASE\n'
for index, name in enumerate(classes):
    command += 'WHEN "COS2015_V1" = \'{}\' THEN {}\n'.format(name, index)
command += 'END'

# Print the command and copy paste into QGis field calculator, creating / editing an integer field called "rasterize".
# https://docs.qgis.org/2.8/en/docs/user_manual/working_with_vector/field_calculator.html
print(command)

# Expected output:
'''
CASE
WHEN "COS2015_V1" = '1.1.1.00.0' THEN 0
WHEN "COS2015_V1" = '1.1.2.00.0' THEN 1
WHEN "COS2015_V1" = '1.2.1.00.0' THEN 2
WHEN "COS2015_V1" = '1.2.2.00.0' THEN 3
WHEN "COS2015_V1" = '1.2.3.00.0' THEN 4
WHEN "COS2015_V1" = '1.2.4.00.0' THEN 5
WHEN "COS2015_V1" = '1.3.1.00.0' THEN 6
WHEN "COS2015_V1" = '1.3.2.00.0' THEN 7
WHEN "COS2015_V1" = '1.3.3.00.0' THEN 8
WHEN "COS2015_V1" = '1.4.1.00.0' THEN 9
WHEN "COS2015_V1" = '1.4.2.01.1' THEN 10
WHEN "COS2015_V1" = '1.4.2.02.0' THEN 11
WHEN "COS2015_V1" = '1.4.2.03.0' THEN 12
WHEN "COS2015_V1" = '2.1.0.00.0' THEN 13
WHEN "COS2015_V1" = '2.1.3.01.1' THEN 14
WHEN "COS2015_V1" = '2.2.1.00.0' THEN 15
WHEN "COS2015_V1" = '2.2.2.00.0' THEN 16
WHEN "COS2015_V1" = '2.2.3.00.0' THEN 17
WHEN "COS2015_V1" = '2.3.1.01.1' THEN 18
WHEN "COS2015_V1" = '2.4.1.00.0' THEN 19
WHEN "COS2015_V1" = '2.4.2.01.1' THEN 20
WHEN "COS2015_V1" = '2.4.3.01.1' THEN 21
WHEN "COS2015_V1" = '2.4.4.00.1' THEN 22
WHEN "COS2015_V1" = '2.4.4.00.2' THEN 23
WHEN "COS2015_V1" = '2.4.4.00.3' THEN 24
WHEN "COS2015_V1" = '2.4.4.00.4' THEN 25
WHEN "COS2015_V1" = '2.4.4.00.5' THEN 26
WHEN "COS2015_V1" = '2.4.4.00.6' THEN 27
WHEN "COS2015_V1" = '2.4.4.00.7' THEN 28
WHEN "COS2015_V1" = '3.1.1.00.1' THEN 29
WHEN "COS2015_V1" = '3.1.1.00.2' THEN 30
WHEN "COS2015_V1" = '3.1.1.00.3' THEN 31
WHEN "COS2015_V1" = '3.1.1.00.4' THEN 32
WHEN "COS2015_V1" = '3.1.1.00.5' THEN 33
WHEN "COS2015_V1" = '3.1.1.00.6' THEN 34
WHEN "COS2015_V1" = '3.1.1.00.7' THEN 35
WHEN "COS2015_V1" = '3.1.2.00.1' THEN 36
WHEN "COS2015_V1" = '3.1.2.00.2' THEN 37
WHEN "COS2015_V1" = '3.1.2.00.3' THEN 38
WHEN "COS2015_V1" = '3.2.1.01.1' THEN 39
WHEN "COS2015_V1" = '3.2.2.00.0' THEN 40
WHEN "COS2015_V1" = '3.3.0.00.0' THEN 41
WHEN "COS2015_V1" = '4.0.0.00.0' THEN 42
WHEN "COS2015_V1" = '5.1.1.00.0' THEN 43
WHEN "COS2015_V1" = '5.1.2.00.0' THEN 44
WHEN "COS2015_V1" = '5.2.1.01.1' THEN 45
WHEN "COS2015_V1" = '5.2.2.01.1' THEN 46
WHEN "COS2015_V1" = '5.2.3.01.1' THEN 47
END
'''
