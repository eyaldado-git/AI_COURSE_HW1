import csv
from Algorithms import *

test_maps = {
    "map12x12": ['SFAFTFFTHHHF',
                 'AFLTFFFFTALF',
                 'LHHLLHHLFTHP',
                 'HALTHAHHAPHF',
                 'FFFTFHFFAHFL',
                 'LLTHFFFAHFAT',
                 'HAAFFALHTATF',
                 'LLLFHFFHTLFH',
                 'FATAFHTTFFAF',
                 'HHFLHALLFTLF',
                 'FFAFFTTAFAAL',
                 'TAAFFFHAFHFG'],
    "map15x15": ['SFTTFFHHHHLFATF',
                 'ALHTLHFTLLFTHHF',
                 'FTTFHHHAHHFAHTF',
                 'LFHTFTALTAAFLLH',
                 'FTFFAFLFFLFHTFF',
                 'LTAFTHFLHTHHLLA',
                 'TFFFAHHFFAHHHFF',
                 'TTFFLFHAHFFTLFP',
                 'TFHLHTFFHAAHFHF',
                 'HHAATLHFFLFFHLH',
                 'FLFHHAALLHLHHAT',
                 'TLHFFLTHFTTFTTF',
                 'AFLTPAFTLHFHFFF',
                 'FFTFHFLTAFLHTLA',
                 'HTFATLTFHLFHFAG'],
    "map20x20" : ['SFFLHFHTALHLFATAHTHT',
                  'HFTTLLAHFTAFAAHHTLFH',
                  'HHTFFFHAFFFFAFFTHHHT',
                  'TTAFHTFHTHHLAHHAALLF',
                  'HLALHFFTHAHHAFFLFHTF',
                  'AFTAFTFLFTTTFTLLTHPF',
                  'LFHFFAAHFLHAHHFHFALA',
                  'AFTFFLTFLFTAFFLTFAHH',
                  'HTTLFTHLTFAFFLAFHFTF',
                  'LLALFHFAHFAALHFTFHTF',
                  'LFFFAAFLFFFFHFLFFAFH',
                  'THHTTFAFLATFATFTHLLL',
                  'HHHAFFFATLLALFAHTHLL',
                  'HLFFFFHFFLAAFTFFPAFH',
                  'HTLFTHFFLTHLHHLHFTFH',
                  'AFTTLHLFFLHTFFAHLAFT',
                  'HAATLHFFFHHHHAFFFHLH',
                  'FHFLLLFHLFFLFTFFHAFL',
                  'LHTFLTLTFATFAFAFHAAF',
                  'FTFFFFFLFTHFTFLTLHFG'],
}

test_envs = {}
for map_name, map_inst in test_maps.items():
    test_envs[map_name] = CampusEnv(map_inst)


DFSG_agent = DFSGAgent()
UCS_agent = UCSAgent()
WAStar_agent = WeightedAStarAgent()
AStar_agent = AStarAgent()

weights = [0.3, 0.7, 0.9]

agents_search_function = [
    DFSG_agent.search,
    UCS_agent.search,
    AStar_agent.search,
]

header = [
    'map',
    'DFS-G cost', 'DFS-G num of expanded nodes',
    'UCS cost', 'UCS  num of expanded nodes',
    'A* cost', 'A*  num of expanded nodes',
    'W-A* (0.3) cost', 'W-A* (0.3) num of expanded nodes',
    'W-A* (0.7) cost', 'W-A* (0.7) num of expanded nodes',
    'W-A* (0.9) cost', 'W-A* (0.9) num of expanded nodes',
]

with open("results.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for env_name, env in test_envs.items():
        data = [env_name]
        for agent in agents_search_function:
            _, total_cost, expanded = agent(env)
            data += [total_cost, expanded]
        for w in weights:
            _, total_cost, expanded = WAStar_agent.search(env, w)
            data += [total_cost, expanded]
        writer.writerow(data)