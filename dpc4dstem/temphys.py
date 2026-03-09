# Relativistic electron parameters	
def compute_electron_wavelength(E0):
	# Compute relativistic electron wavelength for given energy E0 in eV
    return cons.h/np.sqrt(2*cons.m_e*cons.e*E0)\
		/np.sqrt(1 + cons.e*E0/2/cons.m_e/cons.c**2) * 10**10

def compute_interaction_parameter(E0):
    lamb_elec = compute_electron_wavelength(E0)
    return (2*np.pi/lamb_elec/E0)*(cons.m_e*cons.c**2+cons.e*E0)\
		/(2*cons.m_e*cons.c**2+cons.e*E0) # Interaction parameter (rad / (V*A))