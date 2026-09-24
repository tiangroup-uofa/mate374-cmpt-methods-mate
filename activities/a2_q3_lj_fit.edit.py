# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import minimize
    return minimize, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # A2 Q3 · Fitting LJ parameters to quantum cluster energies

    Fit one pair of LJ parameters to the DFT interaction energies of 100
    Ar₃–Ar₇ clusters, then test your chosen parameters on five larger Ar₂₀
    clusters. Coordinates are in Å and energies are in eV, relative to
    separated atoms. All coordinates remain fixed during fitting.

    Complete the marked code in Q3.2 and Q3.6. The cluster-energy calculation
    from A1 and the fitting call are provided.
    """)
    return


@app.cell(hide_code=True)
def embedded_data(np):
    # Coordinates and QM interaction energies copied from the original EXTXYZ
    # datasets in assignments/A2/calculations/. No file download is required.
    small_records = [('ar5_shape3_r4.4', [[12.83545189, 10.15, 10.95], [10.24919678, 13.70967478, 10.95], [6.06454811, 12.35, 10.95], [6.06454811, 7.95, 10.95], [10.24919678, 6.59032522, 10.95]], -0.05281427999999999), ('ar4_shape5_r4.4', [[10.91, 10.47, 10.588], [10.91, 7.83, 6.012], [7.39, 10.47, 6.012], [7.39, 7.83, 10.588]], -0.02959845), ('ar7_shape1_r3.2', [[12.91214683, 10.05, 11.55], [11.03123402, 12.63885438, 11.55], [7.98785317, 11.65, 11.55], [7.98785317, 8.45, 11.55], [11.03123402, 7.46114562, 11.55], [10.19006424, 10.05, 14.27208259], [10.19006424, 10.05, 8.82791741]], 0.00933953999999998), ('ar3_shape4_r3.2', [[8.45, 6.99, 6.0], [11.65, 6.99, 6.0], [11.01, 10.51, 6.0]], 0.006958800000000001), ('ar7_shape3_r3.2', [[13.65, 10.05, 11.55], [12.05, 12.82128129, 11.55], [8.85, 12.82128129, 11.55], [7.25, 10.05, 11.55], [8.85, 7.27871871, 11.55], [12.05, 7.27871871, 11.55], [10.45, 10.05, 11.55]], 0.35422416), ('ar5_distorted_01', [[12.27141326, 11.48044258, 13.45375635], [13.15184318, 9.55280667, 6.84624365], [9.27957443, 6.07077315, 9.48229386], [6.54815682, 8.43086975, 8.65562517], [7.2534565, 14.12922685, 10.53440526]], -0.02006207), ('ar5_shape2_r4.4', [[12.56126984, 10.15, 9.39436508], [9.45, 13.26126984, 9.39436508], [6.33873016, 10.15, 9.39436508], [9.45, 7.03873016, 9.39436508], [9.45, 10.15, 12.50563492]], -0.07979011999999999), ('ar6_shape1_r3.8', [[13.13700577, 9.85, 10.5], [10.45, 12.53700577, 10.5], [10.45, 9.85, 13.18700577], [7.76299423, 9.85, 10.5], [10.45, 7.16299423, 10.5], [10.45, 9.85, 7.81299423]], -0.16563025), ('ar5_shape5_r4.4', [[12.88582276, 10.15, 9.5756709], [9.45, 14.27298731, 9.5756709], [6.01417724, 10.15, 9.5756709], [9.45, 6.02701269, 9.5756709], [9.45, 10.15, 12.3243291]], -0.04118175000000002), ('ar7_shape4_r3.2', [[13.34664333, 10.05, 11.55], [11.13380473, 12.63885438, 11.55], [7.55335667, 11.65, 11.55], [7.55335667, 8.45, 11.55], [11.13380473, 7.46114562, 11.55], [10.14419323, 10.05, 15.55306263], [10.14419323, 10.05, 7.54693737]], -0.05897043000000002), ('ar5_shape4_r4.4', [[11.83156986, 10.15, 10.95], [7.06843014, 12.35, 10.95], [7.06843014, 7.95, 10.95], [8.65614338, 10.15, 15.88980431], [8.65614338, 10.15, 6.01019569]], -0.02683352), ('ar3_shape4_r3.8', [[8.15, 6.66, 6.0], [11.95, 6.66, 6.0], [11.19, 10.84, 6.0]], -0.03095932), ('ar6_shape4_r3.8', [[13.41730148, 9.85, 10.5], [10.45, 12.22384118, 10.5], [10.45, 9.85, 14.35749192], [7.48269852, 9.85, 10.5], [10.45, 7.47615882, 10.5], [10.45, 9.85, 6.64250808]], -0.11849767000000003), ('ar5_distorted_02', [[6.57777847, 12.43948242, 10.42691702], [12.9756655, 9.68165996, 12.32560099], [6.00192443, 8.29566825, 10.1741768], [10.59918719, 7.76051758, 10.58403636], [13.69807557, 8.03690751, 7.97439901]], -0.05170707000000001), ('ar6_shape2_r3.2', [[11.95, 9.85, 8.9], [8.95, 11.58205081, 8.9], [8.95, 8.11794919, 8.9], [11.95, 9.85, 12.1], [8.95, 11.58205081, 12.1], [8.95, 8.11794919, 12.1]], 0.03480849999999999), ('ar7_shape4_r4.4', [[14.43288458, 10.05, 11.55], [11.39023151, 13.60967478, 11.55], [6.46711542, 12.25, 11.55], [6.46711542, 7.85, 11.55], [11.39023151, 6.49032522, 11.55], [10.02951569, 10.05, 17.05421111], [10.02951569, 10.05, 6.04578889]], -0.03808197000000002), ('ar6_shape5_r4.4', [[13.2, 9.85, 8.3], [7.7, 13.66051178, 8.3], [7.7, 6.03948822, 8.3], [13.2, 9.85, 12.7], [7.7, 13.66051178, 12.7], [7.7, 6.03948822, 12.7]], -0.034009979999999995), ('ar5_distorted_08', [[7.18206012, 12.06929211, 12.22489459], [6.90500099, 6.8174216, 10.61248695], [12.79499901, 13.3825784, 11.55506505], [11.7660268, 11.33159596, 9.2407205], [8.48846137, 11.31368736, 8.07510541]], -0.01223312), ('ar7_shape2_r3.2', [[12.7127417, 10.05, 11.55], [10.45, 12.3127417, 11.55], [10.45, 10.05, 13.8127417], [8.1872583, 10.05, 11.55], [10.45, 7.7872583, 11.55], [10.45, 10.05, 9.2872583], [12.7127417, 12.3127417, 13.8127417]], 0.45570538), ('ar7_distorted_09', [[12.20593133, 6.58688468, 9.21804811], [7.65733942, 11.89821062, 10.17816589], [7.3260342, 9.30876508, 7.92248504], [12.5739658, 11.80604224, 6.10658463], [10.12935453, 8.23988767, 12.79341537], [11.17934059, 14.11311532, 8.14175606], [8.42457791, 13.50707731, 7.0311162]], 0.019807820000000004), ('ar7_shape5_r3.2', [[12.88820582, 10.05, 11.55], [10.45, 12.97584698, 11.55], [10.45, 10.05, 13.62247494], [8.01179418, 10.05, 11.55], [10.45, 7.12415302, 11.55], [10.45, 10.05, 9.47752506], [12.88820582, 12.97584698, 13.62247494]], 0.03708224999999998), ('ar6_shape1_r4.4', [[13.56126984, 9.85, 10.5], [10.45, 12.96126984, 10.5], [10.45, 9.85, 13.61126984], [7.33873016, 9.85, 10.5], [10.45, 6.73873016, 10.5], [10.45, 9.85, 7.38873016]], -0.11788395999999998), ('ar3_shape4_r4.4', [[7.85, 6.33, 6.0], [12.25, 6.33, 6.0], [11.37, 11.17, 6.0]], -0.016593560000000007), ('ar4_shape4_r3.2', [[10.49150979, 10.49150979, 9.17198136], [10.49150979, 7.80849021, 7.42801864], [7.80849021, 10.49150979, 7.42801864], [7.80849021, 7.80849021, 9.17198136]], 0.09704972), ('ar3_shape5_r3.8', [[8.71498427, 6.93819294, 6.0], [12.52931493, 6.93819294, 6.0], [7.57068507, 10.56180706, 6.0]], -0.032448580000000005), ('ar6_shape3_r4.4', [[14.85, 9.85, 10.5], [12.65, 13.66051178, 10.5], [8.25, 13.66051178, 10.5], [6.05, 9.85, 10.5], [8.25, 6.03948822, 10.5], [12.65, 6.03948822, 10.5]], -0.06294962000000001), ('ar5_distorted_05', [[12.45599666, 12.46016742, 12.88887759], [7.91442609, 13.75461035, 12.55337374], [9.38724188, 8.53296609, 8.39256944], [7.24400334, 6.44538965, 11.10730773], [8.52490794, 11.3240624, 7.41112241]], 0.03338950999999998), ('ar5_shape2_r3.2', [[11.7127417, 10.15, 9.81862915], [9.45, 12.4127417, 9.81862915], [7.1872583, 10.15, 9.81862915], [9.45, 7.8872583, 9.81862915], [9.45, 10.15, 12.08137085]], 0.23458561), ('ar6_distorted_04', [[13.1529384, 11.01022482, 7.48003598], [9.72057574, 7.19180103, 7.96065877], [8.30088322, 12.50819897, 10.86601276], [13.02918442, 7.25439925, 11.91996402], [6.8470616, 9.23211896, 10.69275196], [6.93500032, 10.97503442, 7.63307933]], -0.06215271), ('ar7_shape3_r4.4', [[14.85, 10.05, 11.55], [12.65, 13.86051178, 11.55], [8.25, 13.86051178, 11.55], [6.05, 10.05, 11.55], [8.25, 6.23948822, 11.55], [12.65, 6.23948822, 11.55], [10.45, 10.05, 11.55]], -0.11907588000000002), ('ar3_shape3_r4.4', [[7.85, 6.0, 6.0], [12.25, 6.0, 6.0], [8.73, 11.5, 6.0]], -0.013035850000000002), ('ar5_shape3_r3.2', [[11.91214683, 10.15, 10.95], [10.03123402, 12.73885438, 10.95], [6.98785317, 11.75, 10.95], [6.98785317, 8.55, 10.95], [10.03123402, 7.56114562, 10.95]], 0.12265311999999999), ('ar7_shape5_r4.4', [[13.802533, 10.05, 11.55], [10.45, 14.0730396, 11.55], [10.45, 10.05, 14.39965305], [7.097467, 10.05, 11.55], [10.45, 6.0269604, 11.55], [10.45, 10.05, 8.70034695], [13.802533, 14.0730396, 14.39965305]], -0.09322459999999999), ('ar6_distorted_06', [[7.49162147, 13.67540267, 9.4324518], [12.60605909, 8.53572932, 11.6200975], [10.49624963, 6.02459733, 8.34811441], [8.13805983, 9.562685, 6.55304021], [8.43875185, 8.96202054, 12.84695979], [7.39394091, 8.10155879, 10.04878559]], -0.015753230000000007), ('ar3_distorted_01', [[9.59710001, 6.79245123, 7.8934589], [7.20611218, 7.64088845, 10.3065411], [12.59388782, 12.60754877, 9.38944502]], -0.009945679999999985), ('ar5_shape1_r4.4', [[11.35525589, 10.15, 10.95], [7.54474411, 12.35, 10.95], [7.54474411, 7.95, 10.95], [8.8149147, 10.15, 14.54258496], [8.8149147, 10.15, 7.35741504]], -0.08649020000000002), ('ar7_shape3_r3.8', [[14.25, 10.05, 11.55], [12.35, 13.34089653, 11.55], [8.55, 13.34089653, 11.55], [6.65, 10.05, 11.55], [8.55, 6.75910347, 11.55], [12.35, 6.75910347, 11.55], [10.45, 10.05, 11.55]], -0.17137525000000003), ('ar7_shape1_r3.8', [[13.37379936, 10.05, 11.55], [11.1402154, 13.12426458, 11.55], [7.52620064, 11.95, 11.55], [7.52620064, 8.15, 11.55], [11.1402154, 6.97573542, 11.55], [10.14132629, 10.05, 14.78247307], [10.14132629, 10.05, 8.31752693]], -0.15621653000000002), ('ar6_shape4_r4.4', [[13.88582276, 9.85, 10.5], [10.45, 12.59865821, 10.5], [10.45, 9.85, 14.96656959], [7.01417724, 9.85, 10.5], [10.45, 7.10134179, 10.5], [10.45, 9.85, 6.03343041]], -0.06319646000000001), ('ar7_distorted_04', [[6.83846549, 12.19287086, 8.17191316], [6.81439414, 10.9882756, 11.98981918], [6.02450901, 9.15578991, 6.54719555], [8.92894208, 8.50712914, 12.35280445], [11.4072673, 10.09204605, 7.04717409], [10.96246302, 12.18153438, 10.90709387], [13.87549099, 8.97053025, 10.72484409]], -0.07796692), ('ar6_shape1_r3.2', [[12.7127417, 9.85, 10.5], [10.45, 12.1127417, 10.5], [10.45, 9.85, 12.7627417], [8.1872583, 9.85, 10.5], [10.45, 7.5872583, 10.5], [10.45, 9.85, 8.2372583]], 0.36408332), ('ar6_shape4_r3.2', [[12.94878019, 9.85, 10.5], [10.45, 11.84902415, 10.5], [10.45, 9.85, 13.74841425], [7.95121981, 9.85, 10.5], [10.45, 7.85097585, 10.5], [10.45, 9.85, 7.25158575]], 0.011929339999999983), ('ar3_distorted_03', [[13.77171808, 10.72689952, 10.06805525], [6.02828192, 11.61233094, 8.13194475], [7.48126091, 7.78766906, 9.41695799]], -0.012207699999999988), ('ar3_shape1_r4.4', [[11.95525589, 8.75, 6.0], [8.14474411, 10.95, 6.0], [8.14474411, 6.55, 6.0]], -0.029870469999999996), ('ar5_shape2_r3.8', [[12.13700577, 10.15, 9.60649712], [9.45, 12.83700577, 9.60649712], [6.76299423, 10.15, 9.60649712], [9.45, 7.46299423, 9.60649712], [9.45, 10.15, 12.29350288]], -0.11506679), ('ar3_shape5_r3.2', [[8.92577623, 7.22426774, 6.0], [12.13784415, 7.22426774, 6.0], [7.96215585, 10.27573226, 6.0]], 0.04839318), ('ar4_shape2_r3.8', [[11.83700577, 9.15, 8.3], [9.15, 11.83700577, 8.3], [6.46299423, 9.15, 8.3], [9.15, 6.46299423, 8.3]], -0.06639703), ('ar6_distorted_05', [[6.79111505, 11.35982352, 12.76054212], [8.50650742, 9.66650266, 6.63945788], [13.20888495, 7.83870437, 6.9611919], [12.98532102, 13.08828476, 10.87820319], [8.83276906, 6.61171524, 11.32586549], [12.52814238, 10.14438449, 11.82568487]], 0.024543610000000007), ('ar6_distorted_01', [[11.05895879, 13.37274247, 10.32002823], [6.39329076, 11.61543412, 9.89962146], [12.71773556, 12.34061009, 7.64826301], [10.70551803, 6.32725753, 11.66538603], [13.60670924, 11.17536568, 10.65305015], [9.6861388, 9.51594437, 11.75173699]], -0.017714829999999987), ('ar5_shape5_r3.2', [[11.94878019, 10.15, 9.95048792], [9.45, 13.14853623, 9.95048792], [6.95121981, 10.15, 9.95048792], [9.45, 7.15146377, 9.95048792], [9.45, 10.15, 11.94951208]], -0.018463869999999993), ('ar4_shape4_r3.8', [[10.74304287, 10.74304287, 9.33547787], [10.74304287, 7.55695713, 7.26452213], [7.55695713, 10.74304287, 7.26452213], [7.55695713, 7.55695713, 9.33547787]], -0.07386936999999999), ('ar3_shape1_r3.8', [[11.69544827, 8.75, 6.0], [8.40455173, 10.65, 6.0], [8.40455173, 6.85, 6.0]], -0.043740680000000004), ('ar3_shape2_r4.4', [[6.03754538, 7.84719771, 6.0], [10.05, 9.65280229, 6.0], [14.06245462, 7.84719771, 6.0]], -0.020437250000000004), ('ar3_shape5_r4.4', [[8.50419231, 6.65211814, 6.0], [12.92078571, 6.65211814, 6.0], [7.17921429, 10.84788186, 6.0]], -0.02065646), ('ar3_distorted_02', [[8.40214752, 11.15960181, 9.11878118], [9.78749569, 8.24039819, 6.74439301], [11.39785248, 8.69178298, 11.45560699]], -0.027618299999999985), ('ar5_shape1_r3.2', [[10.83564065, 10.15, 10.95], [8.06435935, 11.75, 10.95], [8.06435935, 8.55, 10.95], [8.98811978, 10.15, 13.56278906], [8.98811978, 10.15, 8.33721094]], 0.27769114), ('ar5_shape4_r3.2', [[11.18205081, 10.15, 10.95], [7.71794919, 11.75, 10.95], [7.71794919, 8.55, 10.95], [8.87264973, 10.15, 14.54258496], [8.87264973, 10.15, 7.35741504]], -0.06719436000000001), ('ar4_shape3_r3.2', [[11.02408514, 9.15, 7.30048792], [7.27591486, 11.31400712, 7.30048792], [7.27591486, 6.98599288, 7.30048792], [8.52530495, 9.15, 9.29951208]], 0.05950953), ('ar3_shape3_r3.2', [[8.45, 6.75, 6.0], [11.65, 6.75, 6.0], [9.09, 10.75, 6.0]], 0.007320199999999999), ('ar4_shape1_r3.2', [[10.28137085, 10.28137085, 9.43137085], [10.28137085, 8.01862915, 7.16862915], [8.01862915, 10.28137085, 7.16862915], [8.01862915, 8.01862915, 9.43137085]], 0.18352298), ('ar4_shape2_r3.2', [[11.4127417, 9.15, 8.3], [9.15, 11.4127417, 8.3], [6.8872583, 9.15, 8.3], [9.15, 6.8872583, 8.3]], 0.10073533), ('ar3_shape2_r3.2', [[7.13185118, 8.09341652, 6.0], [10.05, 9.40658348, 6.0], [12.96814882, 8.09341652, 6.0]], 0.05232723), ('ar4_distorted_09', [[7.51613105, 8.31720232, 12.55286199], [11.18386895, 11.38279768, 8.77434501], [8.05068154, 10.40144115, 6.74713801], [7.7983193, 10.17989895, 9.87941382]], 0.024347300000000002), ('ar7_shape4_r3.8', [[13.88976395, 10.05, 11.55], [11.26201812, 13.12426458, 11.55], [7.01023605, 11.95, 11.55], [7.01023605, 8.15, 11.55], [11.26201812, 6.97573542, 11.55], [10.08685446, 10.05, 16.30363687], [10.08685446, 10.05, 6.79636313]], -0.08067646), ('ar6_shape5_r3.8', [[12.825, 9.85, 8.6], [8.075, 13.14089653, 8.6], [8.075, 6.55910347, 8.6], [12.825, 9.85, 12.4], [8.075, 13.14089653, 12.4], [8.075, 6.55910347, 12.4]], -0.05671587), ('ar5_shape1_r3.8', [[11.09544827, 10.15, 10.95], [7.80455173, 12.05, 10.95], [7.80455173, 8.25, 10.95], [8.90151724, 10.15, 14.05268701], [8.90151724, 10.15, 7.84731299]], -0.12008542), ('ar3_distorted_09', [[12.66669742, 11.46640298, 6.34732963], [9.34391565, 7.93359702, 7.87242903], [7.13330258, 10.51967367, 11.85267037]], -0.007569009999999987), ('ar7_shape2_r3.8', [[13.13700577, 10.05, 11.55], [10.45, 12.73700577, 11.55], [10.45, 10.05, 14.23700577], [7.76299423, 10.05, 11.55], [10.45, 7.36299423, 11.55], [10.45, 10.05, 8.86299423], [13.13700577, 12.73700577, 14.23700577]], -0.2051675), ('ar4_shape3_r4.4', [[11.72686707, 9.15, 6.9256709], [6.57313293, 12.12550979, 6.9256709], [6.57313293, 6.17449021, 6.9256709], [8.29104431, 9.15, 9.6743291]], -0.03411594000000001), ('ar4_shape5_r3.2', [[10.43, 10.11, 9.964], [10.43, 8.19, 6.636], [7.87, 10.11, 6.636], [7.87, 8.19, 9.964]], 0.008946480000000007), ('ar5_distorted_07', [[8.68742863, 10.23350039, 14.2886224], [12.52987761, 13.25087892, 10.8183097], [12.06896234, 6.94912108, 8.04258395], [10.52916669, 8.6056798, 6.0113776], [7.17012239, 10.06754504, 9.49436061]], 0.06150684999999999), ('ar6_shape3_r3.8', [[14.25, 9.85, 10.5], [12.35, 13.14089653, 10.5], [8.55, 13.14089653, 10.5], [6.65, 9.85, 10.5], [8.55, 6.55910347, 10.5], [12.35, 6.55910347, 10.5]], -0.09841194), ('ar4_shape1_r4.4', [[10.70563492, 10.70563492, 9.85563492], [10.70563492, 7.59436508, 6.74436508], [7.59436508, 10.70563492, 6.74436508], [7.59436508, 7.59436508, 9.85563492]], -0.058134099999999994), ('ar3_distorted_07', [[7.79670221, 10.40434853, 11.57862735], [8.71485378, 10.41104734, 8.55205681], [12.00329779, 8.98895266, 6.62137265]], 0.02048172000000001), ('ar6_distorted_07', [[13.95783185, 12.90197736, 7.57575396], [6.04216815, 8.42271354, 9.47167054], [12.86575098, 7.03870116, 7.17004189], [11.40983028, 11.06639667, 13.35173707], [10.95041444, 6.79802264, 9.67571167], [9.97935747, 9.59315142, 6.04826293]], 0.0009269900000000164), ('ar7_shape2_r4.4', [[13.56126984, 10.05, 11.55], [10.45, 13.16126984, 11.55], [10.45, 10.05, 14.66126984], [7.33873016, 10.05, 11.55], [10.45, 6.93873016, 11.55], [10.45, 10.05, 8.43873016], [13.56126984, 13.16126984, 14.66126984]], -0.14688373999999998), ('ar4_distorted_04', [[8.77505195, 11.26585865, 8.58434677], [10.9510586, 6.02296952, 11.14931671], [12.31532169, 11.19154523, 8.15068329], [6.38467831, 13.67703048, 10.44529191]], -0.030032799999999985), ('ar7_shape1_r4.4', [[13.83545189, 10.05, 11.55], [11.24919678, 13.60967478, 11.55], [7.06454811, 12.25, 11.55], [7.06454811, 7.85, 11.55], [11.24919678, 6.49032522, 11.55], [10.09258833, 10.05, 15.29286356], [10.09258833, 10.05, 7.80713644]], -0.08229467000000001), ('ar3_distorted_04', [[7.78649886, 8.05296014, 12.1922564], [12.01350114, 7.69938772, 7.27524499], [11.03369715, 11.70061228, 6.0077436]], -0.012193249999999989), ('ar5_shape4_r3.8', [[11.50681033, 10.15, 10.95], [7.39318967, 12.05, 10.95], [7.39318967, 8.25, 10.95], [8.76439656, 10.15, 15.21619464], [8.76439656, 10.15, 6.68380536]], -0.06067157000000001), ('ar6_shape2_r3.8', [[12.23125, 9.85, 8.6], [8.66875, 11.90681033, 8.6], [8.66875, 7.79318967, 8.6], [12.23125, 9.85, 12.4], [8.66875, 11.90681033, 12.4], [8.66875, 7.79318967, 12.4]], -0.1340339), ('ar4_shape3_r3.8', [[11.37547611, 9.15, 7.11307941], [6.92452389, 11.71975846, 7.11307941], [6.92452389, 6.58024154, 7.11307941], [8.40817463, 9.15, 9.48692059]], -0.05612756999999999), ('ar7_shape5_r3.8', [[13.34536941, 10.05, 11.55], [10.45, 13.52444329, 11.55], [10.45, 10.05, 14.011064], [7.55463059, 10.05, 11.55], [10.45, 6.57555671, 11.55], [10.45, 10.05, 9.088936], [13.34536941, 13.52444329, 14.011064]], -0.17307624), ('ar6_shape5_r3.2', [[12.45, 9.85, 8.9], [8.45, 12.62128129, 8.9], [8.45, 7.07871871, 8.9], [12.45, 9.85, 12.1], [8.45, 12.62128129, 12.1], [8.45, 7.07871871, 12.1]], 0.04803111), ('ar4_shape4_r4.4', [[10.99457596, 10.99457596, 9.49897437], [10.99457596, 7.30542404, 7.10102563], [7.30542404, 10.99457596, 7.10102563], [7.30542404, 7.30542404, 9.49897437]], -0.04645094999999999), ('ar6_shape3_r3.2', [[13.65, 9.85, 10.5], [12.05, 12.62128129, 10.5], [8.85, 12.62128129, 10.5], [7.25, 9.85, 10.5], [8.85, 7.07871871, 10.5], [12.05, 7.07871871, 10.5]], 0.14956691), ('ar4_distorted_03', [[9.15606513, 6.28989231, 10.43624909], [7.69405814, 10.21758732, 12.54419529], [11.00594186, 8.42678797, 6.75580471], [8.62478515, 13.41010769, 12.17216716]], -0.012075429999999998), ('ar6_shape2_r4.4', [[12.5125, 9.85, 8.3], [8.3875, 12.23156986, 8.3], [8.3875, 7.46843014, 8.3], [12.5125, 9.85, 12.7], [8.3875, 12.23156986, 12.7], [8.3875, 7.46843014, 12.7]], -0.07182332), ('ar4_shape2_r4.4', [[12.26126984, 9.15, 8.3], [9.15, 12.26126984, 8.3], [6.03873016, 9.15, 8.3], [9.15, 6.03873016, 8.3]], -0.04251211999999999), ('ar3_distorted_05', [[7.45852159, 9.01002575, 9.23515588], [12.34147841, 8.0839645, 12.0133348], [11.45970248, 11.3160355, 6.1866652]], -0.004515109999999989), ('ar3_shape2_r3.8', [[6.58469828, 7.97030711, 6.0], [10.05, 9.52969289, 6.0], [13.51530172, 7.97030711, 6.0]], -0.03195948), ('ar3_shape1_r3.2', [[11.43564065, 8.75, 6.0], [8.66435935, 10.35, 6.0], [8.66435935, 7.15, 6.0]], 0.087257), ('ar4_shape5_r3.8', [[10.67, 10.29, 10.276], [10.67, 8.01, 6.324], [7.63, 10.29, 6.324], [7.63, 8.01, 10.276]], -0.055214369999999985), ('ar5_shape3_r3.8', [[12.37379936, 10.15, 10.95], [10.1402154, 13.22426458, 10.95], [6.52620064, 12.05, 10.95], [6.52620064, 8.25, 10.95], [10.1402154, 7.07573542, 10.95]], -0.08296048), ('ar3_distorted_08', [[12.43659454, 7.15022192, 9.3182786], [7.36340546, 12.24977808, 10.48349045], [9.63831003, 7.50722826, 7.71650955]], 0.01651094000000001), ('ar5_shape5_r3.8', [[12.41730148, 10.15, 9.76307941], [9.45, 13.71076177, 9.76307941], [6.48269852, 10.15, 9.76307941], [9.45, 6.58923823, 9.76307941], [9.45, 10.15, 12.13692059]], -0.08186384), ('ar4_distorted_02', [[7.24735966, 11.17915803, 6.04350359], [11.45264034, 12.44761616, 8.43428833], [7.48069319, 8.31307889, 13.25649641], [11.27964777, 7.25238384, 8.17067284]], -0.010251079999999996), ('ar3_shape3_r3.8', [[8.15, 6.375, 6.0], [11.95, 6.375, 6.0], [8.91, 11.125, 6.0]], -0.023549780000000006), ('ar4_shape1_r3.8', [[10.49350288, 10.49350288, 9.64350288], [10.49350288, 7.80649712, 6.95649712], [7.80649712, 10.49350288, 6.95649712], [7.80649712, 7.80649712, 9.64350288]], -0.08159738999999999), ('ar4_distorted_01', [[7.66383111, 10.63104688, 6.19649271], [8.12848024, 12.68014908, 13.10350729], [11.03616889, 9.31702579, 9.94069898], [9.77992702, 7.01985092, 8.43019412]], 0.06917015)]
    large_records = [('ar20_benchmark_01', [[11.49984143, 11.51520991, 11.66520991], [8.87154265, 8.86999173, 11.73684037], [8.87154265, 11.58684037, 9.01999173], [8.84969463, 11.53541067, 14.33451658], [8.84969463, 14.18451658, 11.68541067], [11.38835189, 8.84175639, 8.99175639], [11.51982015, 8.85102277, 14.34074749], [11.51982015, 14.19074749, 9.00102277], [11.54143212, 14.16291916, 14.31291916], [14.10124173, 8.8232684, 11.61986506], [14.10124173, 11.46986506, 8.9732684], [14.18269204, 11.48487055, 14.28975579], [14.18269204, 14.13975579, 11.63487055], [6.19835475, 11.4565383, 11.6065383], [11.45059099, 6.17292666, 11.66170629], [11.45059099, 11.51170629, 6.32292666], [11.53774369, 11.52076996, 16.97707334], [11.53774369, 16.82707334, 11.67076996], [16.80164525, 11.44073329, 11.59073329], [6.57512125, 8.68441494, 8.83441494]], -0.84079017), ('ar20_benchmark_02', [[11.35968699, 11.21795953, 11.53773415], [8.59158717, 8.94075516, 11.49463029], [8.79653078, 11.59928459, 8.88730984], [8.758379, 11.42638006, 14.13108498], [8.71553488, 14.11667556, 11.70258331], [11.20689976, 8.65854983, 8.81612551], [11.40384258, 8.6405214, 14.15776948], [11.3480157, 14.20302308, 8.83154748], [11.3387177, 14.25580198, 14.10157688], [14.11074984, 8.74365014, 11.5937208], [14.0434661, 11.37343231, 8.835159], [14.00816374, 11.23421353, 14.13857539], [14.0891925, 14.1534846, 11.48542932], [6.11614624, 11.17205954, 11.48870759], [11.19861223, 6.12431134, 11.54358185], [11.39870713, 11.51355589, 6.36675302], [11.43473195, 11.46576611, 16.93324698], [11.64330257, 16.87568866, 11.58223171], [16.88385376, 11.22287571, 11.44675945], [6.64934487, 8.46810698, 8.50979905]], -0.7953702300000001), ('ar20_benchmark_03', [[11.64572408, 11.65159558, 11.78940248], [8.97264028, 8.6856992, 11.75632945], [8.74503674, 11.40772324, 9.2697015], [8.88174397, 11.47660228, 14.7790349], [8.7573292, 14.20489542, 11.8885939], [11.15435121, 8.68117354, 8.97240119], [11.53965279, 8.92304496, 14.36651382], [11.41512692, 14.05495912, 8.66093587], [11.52596303, 14.13322509, 14.40928808], [13.81959467, 8.84734359, 11.65261944], [13.83828329, 11.29091652, 9.19633727], [14.04512048, 11.52425506, 14.54335578], [14.00974501, 14.08377403, 12.03162848], [6.14314462, 11.45358439, 11.70130832], [11.17699264, 6.02462909, 11.70321098], [11.36033799, 11.35768487, 6.36014443], [11.31206788, 11.47498037, 16.93985557], [11.58009932, 16.97537091, 11.6101055], [16.85685538, 11.21315247, 11.99759411], [6.62953461, 8.9474865, 8.99204351]], -0.6696173000000001), ('ar20_benchmark_04', [[11.30203657, 11.23999684, 11.35364851], [8.78135463, 8.31972586, 11.54687055], [8.81445769, 11.25679366, 8.73970549], [8.77902539, 11.1870968, 14.66173822], [9.03301389, 13.88154927, 11.59597538], [11.5353544, 8.80241812, 8.73755122], [11.44381701, 8.66149906, 14.13374018], [11.56591422, 13.60085507, 9.17880233], [11.52688791, 13.93611808, 13.77944446], [14.06037403, 8.65317436, 11.85279679], [13.76370419, 11.24254918, 8.67059079], [14.29679738, 11.17822197, 14.27180947], [13.9545976, 13.73207186, 11.50183801], [6.41109743, 11.34044642, 11.6860554], [11.41097348, 6.16310152, 11.67712552], [11.37684463, 11.3002845, 6.19410264], [11.27529127, 11.01295232, 17.10589736], [11.36297842, 16.83689848, 11.35826568], [16.58890257, 11.15868068, 11.28715728], [6.53419068, 8.74713305, 8.95125694]], -0.5510022300000001), ('ar20_benchmark_05', [[11.97485472, 11.64460739, 11.95246762], [9.14913414, 9.20529452, 11.94176217], [9.15768316, 11.73070074, 9.55081694], [9.07994551, 11.84193833, 14.43556223], [8.63652689, 14.13035707, 12.0250304], [12.07241504, 8.84167644, 8.72129108], [12.23501518, 8.84115667, 14.63581918], [11.51234088, 14.73488161, 9.39016471], [11.71391744, 14.532029, 14.52623006], [14.52776141, 9.03205568, 11.42726767], [14.1970356, 11.69934148, 9.52228769], [14.76617257, 11.65763494, 14.29419039], [14.52670609, 14.44874199, 11.42616443], [6.04626536, 11.43053324, 11.65525434], [11.38946849, 6.24120973, 11.62284799], [11.10839874, 12.03502173, 6.03341969], [11.653909, 11.39970295, 17.26658031], [11.73703203, 16.75879027, 11.99594321], [16.95373464, 11.49945272, 11.83308684], [6.33284217, 8.50765654, 8.88340495]], -0.61639127)]

    def as_cluster(record):
        name, coords, energy = record
        return {"config_id": name, "coords": np.array(coords, dtype=float),
                "energy_eV": energy, "n_atoms": len(coords)}

    # Shuffle once with a fixed seed. Larger fitting sets retain smaller sets.
    _order = np.random.default_rng(374).permutation(len(small_records))
    clusters = [as_cluster(small_records[i]) for i in _order]
    ar20_clusters = [as_cluster(record) for record in large_records]
    return ar20_clusters, clusters


@app.cell(hide_code=True)
def supplied_cluster_energy(np):
    def calculate_cluster_energy(coords, epsilon, sigma):
        """Sum all unique LJ pairs, with no cutoff or periodic boundaries."""
        coords = np.asarray(coords, dtype=float)
        i, j = np.triu_indices(len(coords), k=1)
        distances = np.linalg.norm(coords[i] - coords[j], axis=1)
        q = (sigma / distances)**6
        return float(np.sum(4 * epsilon * (q*q - q)))

    return (calculate_cluster_energy,)


@app.cell(hide_code=True)
def residual_definition(mo):
    mo.md(r"""
    ## Q3.1 · Squared residual

    Let $c$ label clusters and $k_c$ be the number of atoms in cluster $c$.
    We compare energies per atom:

    $$
    E(\varepsilon,\sigma)=\sum_{c=1}^{N}
    \left[\frac{E_c^{\mathrm{QM}}}{k_c}
    -\frac{E_c^{\mathrm{LJ}}(\varepsilon,\sigma)}{k_c}\right]^2.
    $$

    What mismatch between the data and the predictions does this residual
    measure? Which two quantities are unknown?
    """)
    return


@app.cell(hide_code=True)
def residual_instructions(mo):
    mo.md(r"""
    ## Q3.2 · Complete the residual loop

    `calculate_cluster_energy` is already implemented using the Assignment 1
    answer. Complete the scaffold `cluster_residual(sigma, epsilon, clusters)`.
    It loops over the clusters and provides the QM and LJ total energies.
    Add the squared per-atom differences and return one scalar sum.

    The check uses three synthetic clusters with arbitrary coordinates and
    dummy reference energies, separate from the DFT data. It tries several
    parameter pairs before unlocking the fit. Copy your working function into
    your answer report.
    """)
    return


@app.cell(hide_code=False)
def student_residual(calculate_cluster_energy):
    def cluster_residual(sigma, epsilon, clusters):
        total = 0.0
        for cluster in clusters:
            coords = cluster["coords"]
            E_QM = cluster["energy_eV"]
            k = cluster["n_atoms"]
            E_LJ = calculate_cluster_energy(coords, epsilon, sigma)
            # COMPLETE BELOW: add the squared per-atom energy difference to total.
            ...
        return total

    return (cluster_residual,)


@app.cell(hide_code=True)
def residual_check(calculate_cluster_energy, cluster_residual, mo, np):
    _rng = np.random.default_rng(4374)
    _dummy = []
    for _k, _energy in [(3, -0.021), (4, -0.038), (5, -0.047)]:
        # Arbitrary, well-separated coordinates with reproducible perturbations.
        _coords = np.column_stack((3.8 * np.arange(_k), np.zeros(_k), np.zeros(_k)))
        _coords += _rng.uniform(-0.2, 0.2, size=(_k, 3))
        _dummy.append({"coords": _coords, "energy_eV": _energy, "n_atoms": _k})
    residual_ok = False
    try:
        for _sigma, _epsilon in [(3.2, 0.009), (3.4, 0.012), (3.6, 0.015)]:
            _value = cluster_residual(_sigma, _epsilon, _dummy)
            if _value is Ellipsis or np.asarray(_value).shape != ():
                raise ValueError("Return one scalar sum.")
            _differences = np.array([
                (c["energy_eV"] - calculate_cluster_energy(c["coords"], _epsilon, _sigma)) / c["n_atoms"]
                for c in _dummy
            ])
            _expected = float(np.dot(_differences, _differences))
            if not np.isfinite(float(_value)) or not np.isclose(float(_value), _expected, rtol=1e-8, atol=1e-12):
                raise ValueError("Check the LJ call, division by the atom count, and accumulation of squared differences.")
        residual_ok = True
        _message = mo.callout(mo.md("The residual passes the three-cluster checks. Fitting is unlocked."), kind="success")
    except Exception as _error:
        _message = mo.callout(mo.md(f"**Fitting locked.** {_error}"), kind="warn")
    _message
    return (residual_ok,)


@app.cell(hide_code=True)
def fitting_instructions(mo):
    mo.md(r"""
    ## Q3.3 · Fit ten clusters

    Read the supplied `fit_LJ` function. In your answer report, briefly describe
    what the `minimize` call does. The optimizer receives `[sigma, epsilon]`; the short function
    inside `minimize` passes these two numbers to your residual.

    Set the slider to **10** and record the fitted parameters and mean absolute
    energy error per atom (MAE).
    """)
    return


@app.cell(hide_code=False)
def supplied_fitting_function(cluster_residual, minimize):
    def fit_LJ(clusters):
        result = minimize(
            lambda values: cluster_residual(values[0], values[1], clusters),
            x0=[3.4, 0.01],  # sigma (Å), epsilon (eV)
            method="Nelder-Mead",
            bounds=[(2.5, 4.5), (1e-6, 0.1)],
            options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 3000},
        )
        return result

    return (fit_LJ,)


@app.cell(hide_code=True)
def dataset_control(mo):
    cluster_count = mo.ui.slider(
        start=10, stop=100, step=10, value=10,
        label="Number of fitting clusters N", show_value=True,
    )
    cluster_count
    return (cluster_count,)


@app.cell(hide_code=True)
def error_support(calculate_cluster_energy, np):
    def energy_errors(clusters, sigma, epsilon):
        differences = np.array([
            calculate_cluster_energy(c["coords"], epsilon, sigma) - c["energy_eV"]
            for c in clusters
        ])
        counts = np.array([c["n_atoms"] for c in clusters])
        return float(np.mean(np.abs(differences))), float(np.mean(np.abs(differences / counts)) * 1000)

    return (energy_errors,)


@app.cell(hide_code=True)
def selected_fit(cluster_count, clusters, energy_errors, fit_LJ, mo, np, residual_ok):
    mo.stop(not residual_ok, mo.md("Complete Q3.2 to calculate the selected fit."))
    selected_clusters = clusters[:int(cluster_count.value)]
    fit_result = fit_LJ(selected_clusters)
    mo.stop(not fit_result.success or not np.all(np.isfinite(fit_result.x)),
            mo.callout(mo.md(f"Fit did not converge: {fit_result.message}"), kind="warn"))
    _sigma, _epsilon = fit_result.x
    _, _mae = energy_errors(selected_clusters, _sigma, _epsilon)
    mo.md(
        f"### Q3.3–Q3.4 · Results for {len(selected_clusters)} clusters\n\n"
        "| Quantity | Value |\n|---|---:|\n"
        f"| σ (Å) | {_sigma:.9g} |\n"
        f"| ε (eV) | {_epsilon:.9g} |\n"
        f"| MAE (meV/atom) | {_mae:.6g} |\n\n"
        f"Optimizer status: {fit_result.message}"
    )
    return fit_result, selected_clusters


@app.cell(hide_code=True)
def dataset_instructions(mo):
    mo.md(r"""
    ## Q3.4 · Increase the dataset

    Use the slider in Q3.3 for **20, 40, 60, 80, and 100 clusters**. Write down
    $\varepsilon$, $\sigma$, and MAE for all six dataset sizes.
    Do the parameters become approximately stable?

    The reported MAE is $\frac{1}{N}\sum_c |(E_c^{\mathrm{LJ}}-E_c^{\mathrm{QM}})/k_c|$,
    converted from eV/atom to meV/atom by multiplying by 1000.
    """)
    return


@app.cell(hide_code=True)
def extrapolation_instructions(mo):
    mo.md(r"""
    ## Q3.5 · Extrapolation to five Ar₂₀ clusters

    Enter your chosen best-fit parameters from Q3.4 and state in your report
    which dataset size you used. These five 20-atom clusters were excluded
    from fitting. The starting zeros leave the calculation locked until you
    enter positive values. For each cluster, the supplied calculation uses
    `E_LJ = calculate_cluster_energy(coords, epsilon, sigma)`.

    Record the mean absolute total-energy difference (eV/cluster) and the mean
    absolute per-atom difference (meV/atom). Compare the per-atom error with
    your corresponding Q3.4 value. What do you observe?
    """)
    return


@app.cell(hide_code=True)
def extrapolation_inputs(mo):
    chosen_sigma = mo.ui.number(value=0.0, step=1e-8, label="Your fitted sigma (Å)")
    chosen_epsilon = mo.ui.number(value=0.0, step=1e-10, label="Your fitted epsilon (eV)")
    mo.vstack([chosen_sigma, chosen_epsilon])
    return chosen_epsilon, chosen_sigma


@app.cell(hide_code=True)
def chosen_values(chosen_epsilon, chosen_sigma, mo, np, residual_ok):
    mo.stop(not residual_ok, mo.md("Complete Q3.2 before testing the larger clusters."))
    test_sigma = float(chosen_sigma.value)
    test_epsilon = float(chosen_epsilon.value)
    mo.stop(not (np.isfinite(test_sigma) and np.isfinite(test_epsilon)
                 and test_sigma > 0 and test_epsilon > 0),
            mo.md("Enter positive, finite fitted parameters from Q3.4."))
    return test_epsilon, test_sigma


@app.cell(hide_code=True)
def extrapolation_results(ar20_clusters, energy_errors, mo, test_epsilon, test_sigma):
    _total_mae, _per_atom_mae = energy_errors(ar20_clusters, test_sigma, test_epsilon)
    mo.md(
        "### Q3.5 · Errors across all five Ar₂₀ clusters\n\n"
        "| Mean absolute difference | Value |\n|---|---:|\n"
        f"| Total energy (eV/cluster) | {_total_mae:.6g} |\n"
        f"| Energy per atom (meV/atom) | {_per_atom_mae:.6g} |"
    )
    return


@app.cell(hide_code=True)
def parity_instructions(mo):
    mo.md(r"""
    ## Q3.6 · Complete the parity-plot loop

    Each point compares a cluster's QM total energy on the horizontal axis
    with its LJ total energy on the vertical axis. Perfect agreement lies
    on $y=x$. Use the same parameters you entered in Q3.5 for both groups.

    Complete only the marked part inside the `for` loop: append the QM energy
    to `qm_energies` and the calculated LJ energy to `lj_energies`. Return the
    two lists in the supplied order. Once the values pass the checks, the plot
    shows all 100 Ar₃–Ar₇ clusters and the five Ar₂₀ clusters with different markers.

    Copy your working `parity_energies` code and the plot into your answer
    report. Do the two groups show different deviations from the diagonal?
    """)
    return


@app.cell(hide_code=False)
def student_parity(calculate_cluster_energy):
    def parity_energies(sigma, epsilon, clusters):
        qm_energies = []
        lj_energies = []
        for cluster in clusters:
            coords = cluster["coords"]
            E_QM = cluster["energy_eV"]
            E_LJ = calculate_cluster_energy(coords, epsilon, sigma)
            # COMPLETE BELOW: append E_QM and E_LJ to their respective lists.
            # These are total energies; do not divide by the atom count.
            ...
        return qm_energies, lj_energies

    return (parity_energies,)


@app.cell(hide_code=True)
def parity_check(ar20_clusters, calculate_cluster_energy, clusters, mo, np, parity_energies, test_epsilon, test_sigma):
    parity_values = None
    try:
        _groups = []
        for _data in [clusters, ar20_clusters]:
            _qm, _lj = parity_energies(test_sigma, test_epsilon, _data)
            _qm, _lj = np.asarray(_qm, dtype=float), np.asarray(_lj, dtype=float)
            if _qm.shape != (len(_data),) or _lj.shape != (len(_data),):
                raise ValueError("Append one energy to each list for every cluster.")
            _expected_qm = np.array([c["energy_eV"] for c in _data])
            _expected_lj = np.array([calculate_cluster_energy(c["coords"], test_epsilon, test_sigma) for c in _data])
            if not (np.all(np.isfinite(_qm)) and np.all(np.isfinite(_lj))
                    and np.allclose(_qm, _expected_qm, rtol=1e-8, atol=1e-12)
                    and np.allclose(_lj, _expected_lj, rtol=1e-8, atol=1e-12)):
                raise ValueError("Check the list order and use total energies in eV, with the supplied parameters.")
            _groups.append((_qm, _lj))
        parity_values = _groups
        _message = mo.md("The energy lists pass the checks.")
    except Exception as _error:
        _message = mo.callout(mo.md(f"**Parity plot locked.** {_error}"), kind="warn")
    _message
    return (parity_values,)


@app.cell(hide_code=True)
def parity_plot(mo, np, parity_values, plt):
    mo.stop(parity_values is None)
    _fig, _ax = plt.subplots(figsize=(6, 6), layout="constrained")
    for (_qm, _lj), _label, _marker in zip(parity_values, ["100 Ar₃–Ar₇ clusters", "5 Ar₂₀ clusters"], ["o", "^"]):
        _ax.scatter(_qm, _lj, label=_label, marker=_marker, alpha=0.8)
    _all = np.concatenate([values for group in parity_values for values in group])
    _pad = max(float(np.ptp(_all)) * 0.06, 0.01)
    _limits = (float(_all.min()) - _pad, float(_all.max()) + _pad)
    _ax.plot(_limits, _limits, "--", color="0.4", label="y = x")
    _ax.set(xlim=_limits, ylim=_limits, xlabel="QM total interaction energy (eV)",
            ylabel="LJ total interaction energy (eV)", title="Q3.6 · Parity plot")
    _ax.set_aspect("equal", adjustable="box")
    _ax.legend()
    _ax.grid(alpha=0.2)
    plt.close(_fig)
    _fig
    return


if __name__ == "__main__":
    app.run()
