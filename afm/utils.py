import numpy as np
def false_alarm_rate(fault_free_spe :np.ndarray, seuil:float)->tuple[int,float]:
    """
    Function to calculate the false alarm rate for NOC data

    Args :
      fault_free_spe : SPE of the NOC data
      seuil : detection limit

    Returns :
      false_positive : number of false positive samples
      false_positive_rate : false_alarm_rate (%)
     """
    false_positive=0
    for i in range(fault_free_spe.shape[0]):
        if fault_free_spe[i]> seuil:
            false_positive+=1
    false_positive_rate = round(false_positive*100/fault_free_spe.shape[0],2)
    print('Number of samples : ', fault_free_spe.shape[0])
    print('Number of false positive samples : ',false_positive)
    print(f'Percentage of false positive samples :{false_positive_rate}%')
    return false_positive, false_positive_rate

def fault_detection_rate(faulty_spe :np.ndarray, seuil)->tuple[int,float]:
    """
    Function to calculate the fault detection rate

    Args :
      faulty_spe : SPE of the faulty data
      seuil : detection limit

    Returns :
      detected_fault : number of detected faulty samples
      fault_detection_r : fault detection rate (%)
     """
    detected_fault=0
    for i in range(faulty_spe.shape[0]):
        if faulty_spe[i]> seuil:
            detected_fault+=1
    fault_detection_r = round((detected_fault*100)/faulty_spe.shape[0],2)
    print('Number of faulty samples : ', faulty_spe.shape[0])
    print('Number of detected faulty samples : ',detected_fault)
    print(f'Percentage of detected faulty_samples :{fault_detection_r}%')
    return detected_fault, fault_detection_r

def false_discovery_rate(n_false_positive:int,n_detected_fault:int):
    '''
    Function for calculating the false dicovery rate

    Args:
        n_false_positive : number of false positive samples
        n_detected_fault : number of detected faulty samples
    '''
    print(f'Percentage of false discoveries :{round(n_false_positive*100/(n_false_positive+n_detected_fault),2)}%')