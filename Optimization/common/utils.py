import numpy as np
from NanoParticleTools.species_data.species import Dopant
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics

def emsInteg(x,y,startWav, endWav):
    sum_int = 0
    for _x, _y in zip(x, y):
        if ((_x>-endWav) and (_x<-startWav)):
            sum_int += _y
    return sum_int

def absInteg(x,y,startWav, endWav):
    sum_int = 0
    for _x, _y in zip(x, y):
        if ((_x<endWav) and (_x>startWav)):
            sum_int += _y
    return sum_int

def get_spectrum(doc):
    dndt = doc['data']['output']['summary'] # dndt = docs
    accumulated_dndt = {}
    for interaction in dndt:
        interaction_id = interaction[0]
        if interaction_id not in accumulated_dndt:
            accumulated_dndt[interaction_id] = []
        accumulated_dndt[interaction_id].append(interaction)
    avg_dndt = []
    for interaction_id in accumulated_dndt:

        arr = accumulated_dndt[interaction_id][-1][:-4]

        _dndt = [_arr[-4:] for _arr in accumulated_dndt[interaction_id]]

        while len(_dndt) < 1:
            _dndt.append([0 for _ in range(4)])

        mean = np.mean(_dndt, axis=0)
        std = np.std(_dndt, axis=0)
        arr.extend([mean[0], std[0], mean[1], std[1], mean[2], std[2], mean[3], std[3],])
        avg_dndt.append(arr)
    
    x = []
    y = []
    dopants = [Dopant(key, val) for key, val in doc['data']['overall_dopant_concentration'].items()]
    for interaction in [_d for _d in avg_dndt if _d[8] == 'Rad']:
        # print(interaction)
        species_id = interaction[2]
        left_state_1 = interaction[4]
        right_state_1 = interaction[6]
        ei = dopants[species_id].energy_levels[left_state_1]
        ef = dopants[species_id].energy_levels[right_state_1]

        de = ef.energy-ei.energy
        wavelength = (299792458*6.62607004e-34)/(de*1.60218e-19/8065.44)*1e9
        # print(left_state_1, right_state_1, wavelength)
        x.append(wavelength)
        y.append(interaction[10])
    return x, y

def get_spectrum_no_data(doc):
    dndt = doc['output']['summary'] # dndt = docs
    accumulated_dndt = {}
    for interaction in dndt:
        interaction_id = interaction[0]
        if interaction_id not in accumulated_dndt:
            accumulated_dndt[interaction_id] = []
        accumulated_dndt[interaction_id].append(interaction)
    avg_dndt = []
    for interaction_id in accumulated_dndt:

        arr = accumulated_dndt[interaction_id][-1][:-4]

        _dndt = [_arr[-4:] for _arr in accumulated_dndt[interaction_id]]

        while len(_dndt) < 1:
            _dndt.append([0 for _ in range(4)])

        mean = np.mean(_dndt, axis=0)
        std = np.std(_dndt, axis=0)
        arr.extend([mean[0], std[0], mean[1], std[1], mean[2], std[2], mean[3], std[3],])
        avg_dndt.append(arr)
    
    x = []
    y = []
    dopants = [Dopant(key, val) for key, val in doc['overall_dopant_concentration'].items()]
    for interaction in [_d for _d in avg_dndt if _d[8] == 'Rad']:
        # print(interaction)
        species_id = interaction[2]
        left_state_1 = interaction[4]
        right_state_1 = interaction[6]
        ei = dopants[species_id].energy_levels[left_state_1]
        ef = dopants[species_id].energy_levels[right_state_1]

        de = ef.energy-ei.energy
        wavelength = (299792458*6.62607004e-34)/(de*1.60218e-19/8065.44)*1e9
        # print(left_state_1, right_state_1, wavelength)
        x.append(wavelength)
        y.append(interaction[10])
    return x, y
def get_int(doc, spec_range):
    x, y = get_spectrum(doc)
    return emsInteg(x, y, spec_range[0], spec_range[1])

def get_qe(doc, total_range, absorption_range):
    x, y = get_spectrum(doc)
    if absInteg(x,y,absorption_range[0], absorption_range[1]) == 0:
        qe =0
    else:
        qe = emsInteg(x,y,total_range[0], total_range[1])/absInteg(x,y,absorption_range[0], absorption_range[1])
    return qe

def get_int_no_data(doc, spec_range):
    x, y = get_spectrum_no_data(doc)
    return emsInteg(x, y, spec_range[0], spec_range[1])

def get_qe_no_data(doc, total_range, absorption_range):
    x, y = get_spectrum_no_data(doc)
    if absInteg(x,y,absorption_range[0], absorption_range[1]) == 0:
        qe =0
    else:
        qe = emsInteg(x,y,total_range[0], total_range[1])/absInteg(x,y,absorption_range[0], absorption_range[1])
    return qe

def get_populations_from_doc(doc):
    x_list=np.array(doc['data']['output']['x_populations'])
    y_list=np.array(doc['data']['output']['y_overall_populations'])
    return x_list, y_list

def get_averaged_population(doc):
    _, y_list = get_populations_from_doc(doc)
    population = np.average(y_list,0)
    return population

def get_dopant_conc(doc):
    from NanoParticleTools.inputs import constants
    dopants = [Dopant(key, val) for key, val in doc['data']['overall_dopant_concentration'].items()]
    total_n_levels = sum([dopant.n_levels for dopant in dopants])
    concentration = np.zeros(total_n_levels)
    for dopant_index, dopant in enumerate(dopants):
        for i in range(dopant.n_levels):
            combined_i = int(sum([dopant.n_levels for dopant in dopants[:dopant_index]]) + i)
            concentration[combined_i] = doc['data']['overall_dopant_concentration'][dopant.symbol]
    return concentration

def get_wavelength_matrix(doc):
    from NanoParticleTools.inputs import constants
    dopants = [Dopant(key, val) for key, val in doc['data']['overall_dopant_concentration'].items()]
    total_n_levels = sum([dopant.n_levels for dopant in dopants])
    energy_gaps = np.zeros((total_n_levels, total_n_levels))
    wavelength_matrix = np.zeros((total_n_levels, total_n_levels))

    for dopant_index, dopant in enumerate(dopants):
        for i in range(dopant.n_levels):
            combined_i = int(sum([dopant.n_levels for dopant in dopants[:dopant_index]]) + i)
            for j in range(dopant.n_levels):
                combined_j = int(sum([dopant.n_levels for dopant in dopants[:dopant_index]]) + j)
                energy_gap = dopant.energy_levels[j].energy - dopant.energy_levels[i].energy
                energy_gaps[combined_i, combined_j] = energy_gap
                if energy_gap ==0:
                    wavelength_matrix[combined_i, combined_j] = 0
                else:
                    wavelength_matrix[combined_i, combined_j] = 1e7 / energy_gap
    return wavelength_matrix

def get_wavelength_selection_matrix(wavelength_matrix, min_wav, max_wav):
    wavelength_selection_matrix = np.zeros_like(wavelength_matrix)
    for i, start_level in enumerate(wavelength_matrix):
        for j, wavelength in enumerate(start_level):
            if  (wavelength < -min_wav) and (wavelength > -max_wav):
                wavelength_selection_matrix[i,j] = 1
            else:
                wavelength_selection_matrix[i,j] = 0
    return wavelength_selection_matrix

def get_PopxRate(doc, MIN_WAV, MAX_WAV):
    dopants = [Dopant(key, val) for key, val in doc['data']['overall_dopant_concentration'].items()]
    sk = SpectralKinetics(dopants)
    rad_const_matrix = sk.radiative_rate_matrix
    population = get_averaged_population(doc)
    dopant_conc = get_dopant_conc(doc)*100
    weighted_population = population*dopant_conc
    rad_rate = rad_const_matrix * weighted_population[:,None]
    wavelength_matrix = get_wavelength_matrix(doc)
    wavelength_selection_matrix = get_wavelength_selection_matrix(wavelength_matrix, MIN_WAV, MAX_WAV)
    return sum(sum(wavelength_selection_matrix * rad_rate))
    