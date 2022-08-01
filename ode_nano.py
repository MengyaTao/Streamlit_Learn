import numpy as np

from advective_processes_nano import dryDepAir
from advective_processes_nano import wetDepAir
from advective_processes_nano import odeHetagg
from advective_processes_nano import airAdvection
from advective_processes_nano import dryDepAer
from advective_processes_nano import wetDep
from advective_processes_nano import waterAdv
from advective_processes_nano import odeDissolution
from advective_processes_nano import sedDeposition
from advective_processes_nano import resuspensionSed
from advective_processes_nano import burial
from advective_processes_nano import aerosolResuspension
from advective_processes_nano import windErosion
from advective_processes_nano import erosion
from advective_processes_nano import soilwaterPartition
from advective_processes_nano import runoff
from advective_processes_nano import vertFlow
from advective_processes_nano import horiFlow

#################################################################
#
#   CLiCC Nanomaterials F&T Model Developed by Dr. Garner
#	Model original devloped in MATLAB
#   Date: December 26th, 2016
#   Converted to Python by Jill Farnsworth
#   Edited Jan 5, 2022 by Dr. Kendra Garner
#
#################################################################

def ode_nano(t,f,i,V,presence,env,climate,ENM,release,bgConc,DIS,time):
	# %   Differential equation solver for ENM mass in all compartments
	# %   t is time, f is the mass by compartment and day, i is the iteration in
	# %   the for loop - so the time step, V is the volume vector
	# Note: i in Python should be one less than i in Matlab, since Python is zero indexed

	# %% Process calculations

	# % Air
	if presence['air']==1:
		# % dry deposition from air
		dryDepositionAir = dryDepAir(ENM['density'],env['airP'],env['dynViscAir'],(ENM['radiusENMagg']*(10**-9)),env['airH'])*f[0]
		# % wet deposition from air
		wetDepositionAir = wetDepAir(climate['precip'][i],env['scavengingENM'],env['area'],V[0])*f[0]
		# % heteroaggregation in air (pseudofirst order rate constant)
		heteroaggregationAirAer = odeHetagg(ENM['khetA'],env['aerC'])*f[0]
		# % advection in air
		advectionAir = airAdvection(climate['windspeed'][i],env['area'],env['airH'],V[0],np.true_divide(f[0],V[0]),ENM['density'])*f[0]
	else:
		dryDepositionAir = 0
		wetDepositionAir = 0
		heteroaggregationAirAer = 0
		advectionAir = 0

	# % Aerosols
	if presence['aer']==1:
		# % dry deposition of aerosols
		dryDepositionAer = dryDepAer(env['aerP'],env['airP'],env['dynViscAir'],env['radiusParticlesAer'],env['airH'])*f[1]
		# % wet deposition of aerosols
		wetDepositionAer = wetDep(climate['precip'][i],V[0],env['scavenging'],env['area'])*f[1]
		# % advection of aerosols
		advectionAer = airAdvection(climate['windspeed'][i],env['area'],env['airH'],V[0],env['aerC'],env['aerP'])*f[1]
	else:
		dryDepositionAer = 0
		wetDepositionAer = 0
		advectionAer = 0

    # % River water
	if presence['rw']==1:
		# % sedimentation of free ENMs
		sedimentationRW = np.true_divide(ENM['ksedRW'],env['riverD_max'])*f[2]
		# % heteroaggregation in river water with suspended sediment
		heteroaggregationRW = odeHetagg(ENM['khetRW'],env['riverssC'])*f[2]
		# % advective flow
		advectionRW = waterAdv(climate['flow1'][i],V[2],np.true_divide(f[2],V[2]),ENM['density'])*f[2]
		# % Dissolution in river water
		dissolutionRW = odeDissolution(DIS['percfitaRW'],DIS['percfitbRW'],f[2],f[19],ENM['kdisFW'],V[2])
	else:
		sedimentationRW = 0
		heteroaggregationRW = 0
		advectionRW = 0
		dissolutionRW = 0
		
    # % River water Suspended Sediment
	if presence['rSS']==1:
		# % deposition of river water suspended sediment
		sedimentationRWSS = sedDeposition(env['riverssP'],env['riverP'],env['dynViscRiver'],
										  env['radiusParticlesRiver'],env['riverD_max'])*f[3]
		# % advection of river water suspended sediment
		advectionRWSS = waterAdv(climate['flow1'][i],V[2],env['freshssC'],env['freshssP'])*f[3]
	else:
		sedimentationRWSS = 0
		advectionRWSS = 0

	# % River water Sediment
	if presence['rSed']==1:
		# % resuspension from river water sediment
		resuspensionRWSed = resuspensionSed(env['sedRWA'],env['resuspensionRateRiver'],V[4])*f[4]
		# % burial in river water sediment
		burialRWSed = burial(env['sedRWA'],env['burialRateRiver'],V[4])*f[4]
		# % advection of sediment
		advectionRWSed = waterAdv(climate['flow1'][i],V[4],f[4]/V[4],env['sedRWP'])*env['riveradvfrac']*f[4]
		# % dissolution in sediment
		dissolutionRWSed = odeDissolution(DIS['percfitaRW'],DIS['percfitbRW'],f[4],f[20],ENM['kdisRWsed'],V[4])
	else:
		resuspensionRWSed = 0
		burialRWSed = 0
		advectionRWSed = 0
		dissolutionRWSed = 0
		
		
	# % Freshwater
	if presence['fw']==1:
		# % sedimentation of free ENMs
		sedimentationFW = np.true_divide(ENM['ksedFW'],env['freshwD'])*f[5]
		# % heteroaggregation in freshwater with suspended sediment
		heteroaggregationFW = odeHetagg(ENM['khetFW'],env['freshssC'])*f[5]
		# % advective flow
		advectionFW = waterAdv(climate['flow2'][i],V[5],np.true_divide(f[5],V[5]),ENM['density'])*f[5]
		# % Dissolution in freshwater
		dissolutionFW = odeDissolution(DIS['percfitaFW'],DIS['percfitbFW'],f[5],f[21],ENM['kdisFW'],V[5])
	else:
		sedimentationFW = 0
		heteroaggregationFW = 0
		advectionFW = 0
		dissolutionFW = 0

	# % Freshwater Suspended Sediment
	if presence['fSS']==1:
		# % deposition of freshwater suspended sediment
		sedimentationFWSS = sedDeposition(env['freshssP'],env['freshwP'],env['dynViscFW'],env['radiusParticlesFW'],env['freshwD'])*f[6]
		# % advection of freshwater suspended sediment
		advectionFWSS = waterAdv(climate['flow2'][i],V[5],env['freshssC'],env['freshssP'])*f[6]
	else:
		sedimentationFWSS = 0
		advectionFWSS = 0

	# % Freshwater Sediment
	if presence['fSed']==1:
		# % resuspension from freshwater sediment
		resuspensionFWSed = resuspensionSed(env['sedFWA'],env['resuspensionRateFW'],V[7])*f[7]
		# % burial in freshwater sediment
		burialFWSed = burial(env['sedFWA'],env['burialRateFW'],V[7])*f[7]
		# % advection of sediment
		advectionFWSed = waterAdv(climate['flow2'][i],V[7],f[7]/V[7],env['sedFWP'])*env['fwadvfrac']*f[7]
		# % dissolution in sediment
		dissolutionFWSed = odeDissolution(DIS['percfitaFW'],DIS['percfitbFW'],f[7],f[22],ENM['kdisFWsed'],V[7])
	else:
		resuspensionFWSed = 0
		burialFWSed = 0
		advectionFWSed = 0
		dissolutionFWSed = 0

	# % Seawater
	if presence['sw']==1:
		# % sedimentation of free ENMs in seawater
		sedimentationSW = (np.true_divide(ENM['ksedSW'],env['seawD']))*f[8]
		# % heteroaggregation with suspended sediment in seawater
		heteroaggregationSW = odeHetagg(ENM['khetSW'],env['seassC'])*f[8]
		# % aerosolization of particles from seawater to aerosols or air
		aerosolizationSW = aerosolResuspension(climate['windspeed'][i],env['coastalA'],ENM['enrichFactor'],env['seawD'],env['seawV'])*f[8]
		# % advection out of the coastal marine to larger system
		advectionSW = waterAdv(climate['flow2'][i],V[8],np.true_divide(f[8],V[8]),ENM['density'])*f[8]
		# % dissolution in marine
		dissolutionSW = odeDissolution(DIS['percfitaSW'],DIS['percfitbSW'],f[8],f[23],ENM['kdisSW'],V[8])
	else:
		sedimentationSW = 0
		heteroaggregationSW = 0
		aerosolizationSW = 0
		advectionSW = 0
		dissolutionSW = 0

	# % Seawater suspended sediment
	if presence['sSS']==1:
		# % deposition of suspended sediment
		sedimentationSWSS = sedDeposition(env['seassP'],env['seawP'],env['dynViscSW'],env['radiusParticlesSW'],env['seawD'])*f[9]
		# % advective flow
		advectionSWSS = waterAdv(climate['flow2'][i],V[8],env['seassC'],env['seassP'])*f[9]
	else:
		sedimentationSWSS = 0
		advectionSWSS = 0

	# % Seawater sediment
	if presence['sSed']==1:
		# % resuspension from marine sediment
		resuspensionSWSed = resuspensionSed(env['sedSWA'],env['resuspensionRateSW'],V[10])*f[10]
		# % burial in marine sediment
		burialSWSed = burial(env['sedSWA'],env['burialRateSW'],V[10])*f[10]
		# % advection of sediment
		advectionSWSed = waterAdv(climate['flow2'][i],V[10],f[10]/V[10],env['sedSWP'])*env['swadvfrac']*f[10]
		# % dissolution in sediment
		dissolutionSWSed = odeDissolution(DIS['percfitaSW'],DIS['percfitbSW'],f[10],f[24],ENM['kdisSWsed'],V[10])
	else:
		resuspensionSWSed = 0
		burialSWSed = 0
		advectionSWSed = 0
		dissolutionSWSed = 0

	# % Soil 1
	if presence['soil1']==1:
		# % wind erosion
		windErosionSoil1 = windErosion(climate['windspeed'][i],climate['precip'][i], \
									   env['roughness1'],env['Kconstant1'],env['airP'],env['soilA1'],\
									   env['A1'],env['TSV1'],env['TSVmin1'],env['z_wind1'],env['percWind1'],\
									   env['windConstant1'],env['percUncovered1'],env['percSuspended1'],env['soilP1'],V[11])*f[11]
		# % solid soil erosion
		solidErosionSoil1 = erosion(climate['precip'][i],env['Kfact1'],env['lenslope1'],\
									env['cropManageFactor1'],env['supportFactor1'],env['soilA1'],env['soilP1'])*np.true_divide(f[11],V[11])
		# % loss by partitioning to soil water
		# [0] calls first value in function
		soil2soilwater1 = soilwaterPartition(f[11],f[12],ENM['elutionS1'],1)[0]
	else:
		windErosionSoil1 = 0
		solidErosionSoil1 = 0
		soil2soilwater1 = 0

	# % Soil Water 1
	if presence['soilW1']==1:
		# % runoff
		runoffSoil1 = runoff(climate['precip'][i],env['CN1'],env['soilA1'], V[11])*f[12]
		# % infiltration
		k_infil1, infiltraSoil1 = vertFlow(climate['precip'][i], env['CN1'], climate['evap'][i], env['FC1'], env['soilWC1'],
										   env['soilV1'], env['soilA1'])
		infiltraSoil1 = infiltraSoil1*f[12]
		# % loss to partitioning to soil solids
		# [1] calls second value in function
		soilwater2soil1 = soilwaterPartition(f[11],f[12],ENM['elutionS1'],2)[1]
		# % dissolution
		dissolutionSoil1 = odeDissolution(DIS['percfitaGW1'],DIS['percfitbGW1'],f[12],f[25],ENM['kdisS1'],V[12])
	else:
		runoffSoil1 = 0
		infiltraSoil1 = 0
		soilwater2soil1 = 0
		dissolutionSoil1 = 0

	# % Soil 2
	if presence['soil2']==1:
		# % wind erosion
		windErosionSoil2 = windErosion(climate['windspeed'][i],climate['precip'][i], env['roughness2'],\
									   env['Kconstant2'],env['airP'],env['soilA2'],env['A2'],env['TSV2'],\
									   env['TSVmin2'],env['z_wind2'],env['percWind2'],env['windConstant2'],\
									   env['percUncovered2'],env['percSuspended2'],env['soilP2'],V[13])*f[13]
		# % solid soil erosion
		solidErosionSoil2 = erosion(climate['precip'][i],env['Kfact2'],env['lenslope2'],\
									env['cropManageFactor2'],env['supportFactor2'],env['soilA2'],env['soilP2'])*np.true_divide(f[13],V[13])
		# % loss by partitioning to soil water
		soil2soilwater2 = soilwaterPartition(f[13],f[14],ENM['elutionS2'],1)[0]
	else:
		windErosionSoil2 = 0
		solidErosionSoil2 = 0
		soil2soilwater2 = 0

	# % Soil Water 2
	if presence['soilW2']==1:
		# % runoff
		runoffSoil2 = runoff(climate['precip'][i],env['CN2'],env['soilA2'],V[13])*f[14]
		# % infiltration
		k_infil2, infiltraSoil2 = vertFlow(climate['precip'][i], env['CN2'], climate['evap'][i], env['FC2'], env['soilWC2'],
										   env['soilV2'], env['soilA2'])
		infiltraSoil2 = infiltraSoil2*f[14]
		# % loss to partitioning to soil solids
		soilwater2soil2 = soilwaterPartition(f[13],f[14],ENM['elutionS2'],2)[1]
		# % dissolution
		dissolutionSoil2 = odeDissolution(DIS['percfitaGW2'],DIS['percfitbGW2'],f[14],f[26],ENM['kdisS2'],V[14])
	else:
		runoffSoil2 = 0
		infiltraSoil2 = 0
		soilwater2soil2 = 0
		dissolutionSoil2 = 0

	# % Soil 3
	if presence['soil3']==1:
		# % wind erosion
		windErosionSoil3 = windErosion(climate['windspeed'][i],climate['precip'][i], env['roughness3'],\
									   env['Kconstant3'],env['airP'],env['soilA3'],env['A3'],env['TSV3'],\
									   env['TSVmin3'],env['z_wind3'],env['percWind3'],env['windConstant3'],\
									   env['percUncovered3'],env['percSuspended3'],env['soilP3'],V[15])*f[15]
		# % solid soil erosion
		solidErosionSoil3 = erosion(climate['precip'][i],env['Kfact3'],env['lenslope3'],\
									env['cropManageFactor3'],env['supportFactor3'],env['soilA3'],env['soilP3'])*np.true_divide(f[15],V[15])
		# % loss by partitioning to soil water
		soil2soilwater3 = soilwaterPartition(f[15],f[16],ENM['elutionS3'],1)[0]
	else:
		windErosionSoil3 = 0
		solidErosionSoil3 = 0
		soil2soilwater3 = 0

	# % Soil Water 3
	if presence['soilW3']==1:
		# % runoff
		runoffSoil3 = runoff(climate['precip'][i],env['CN3'],env['soilA3'],V[15])*f[16]
		# % infiltration
		k_infil3, infiltraSoil3 = vertFlow(climate['precip'][i], env['CN3'], climate['evap'][i], env['FC3'], env['soilWC3'],
										   env['soilV3'], env['soilA3'])
		infiltraSoil3 = infiltraSoil3*f[16]
		# % loss to partitioning to soil solids
		soilwater2soil3 = soilwaterPartition(f[15],f[16],ENM['elutionS3'],2)[1]
		# % dissolution
		dissolutionSoil3 = odeDissolution(DIS['percfitaGW3'],DIS['percfitbGW3'],f[16],f[27],ENM['kdisS3'],V[16])
	else:
		runoffSoil3 = 0
		infiltraSoil3 = 0
		soilwater2soil3 = 0
		dissolutionSoil3 = 0

	# % Soil 4
	if presence['soil4']==1:
		# % wind erosion
		windErosionSoil4 = windErosion(climate['windspeed'][i],climate['precip'][i], env['roughness4'],env['Kconstant4'],\
									   env['airP'],env['soilA4'],env['A4'],env['TSV4'],env['TSVmin4'],env['z_wind4'],\
									   env['percWind4'],env['windConstant4'],env['percUncovered4'],env['percSuspended4'],\
									   env['soilP4'],V[17])*f[17]
		# % solid soil erosion
		solidErosionSoil4 = erosion(climate['precip'][i],env['Kfact4'],env['lenslope4'],\
									env['cropManageFactor4'],env['supportFactor4'],env['soilA4'],env['soilP4'])*np.true_divide(f[17],V[17])
		# % loss by partitioning to soil water
		soil2soilwater4 = soilwaterPartition(f[17],f[18],ENM['elutionS4'],1)[0]
	else:
		windErosionSoil4 = 0
		solidErosionSoil4 = 0
		soil2soilwater4 = 0

	# % Soil Water 4
	if presence['soilW4']==1:
		# % runoff
		runoffSoil4 = runoff(climate['precip'][i],env['CN4'],env['soilA4'],V[17])*f[18]
		# % infiltration
		k_infil4, infiltraSoil4 = vertFlow(climate['precip'][i], env['CN4'], climate['evap'][i], env['FC4'], env['soilWC4'],
										   env['soilV4'], env['soilA4'])
		infiltraSoil4 = infiltraSoil4*f[18]
		# % loss to partitioning to soil solids
		soilwater2soil4 = soilwaterPartition(f[17],f[18],ENM['elutionS4'],2)[1]
		# % dissolution
		dissolutionSoil4 = odeDissolution(DIS['percfitaGW4'],DIS['percfitbGW4'],f[18],f[28],ENM['kdisS4'],V[18])
	else:
		runoffSoil4 = 0
		infiltraSoil4 = 0
		soilwater2soil4 = 0
		dissolutionSoil4 = 0
	
	# % Dissolved RW mass transport
	if presence['rw'] == 1:
		advectionRWDis = waterAdv(climate['flow1'][i],V[2],f[19]/V[2],env['riverP'])*f[19]
	else:
		advectionRWDis = 0
	
	# % Dissolved RW sediment mass transport
	if presence['rw'] == 1:
		advectionRWSedDis = waterAdv(climate['flow1'][i],V[4],f[20]/V[4],env['sedRWP'])*env['riveradvfrac']*f[20]
	else:
		advectionRWSedDis = 0
		
	# % Dissolved FW mass transport
	if presence['fw'] == 1:
		advectionFWDis = waterAdv(climate['flow2'][i],V[5],f[21]/V[5],env['freshwP'])*f[21]
	else:
		advectionFWDis = 0
	
	# % Dissolved FW sediment mass transport
	if presence['fw'] == 1:
		advectionFWSedDis = waterAdv(climate['flow2'][i],V[7],f[22]/V[7],env['sedFWP'])*env['fwadvfrac']*f[22]
	else:
		advectionFWSedDis = 0
		
	# % Dissolved SW mass transport
	if presence['sw'] == 1:
		advectionSWDis = waterAdv(climate['flow2'][i],V[8],f[23]/V[8],env['seawP'])*f[23]
	else:
		advectionSWDis = 0
	
	# % Dissolved SW sediment mass transport
	if presence['sw'] == 1:
		advectionSWSedDis = waterAdv(climate['flow2'][i],V[10],f[24]/V[10],env['sedSWP'])*env['fwadvfrac']*f[24]
	else:
		advectionSWSedDis = 0
	
	# % Dissolved Soil Water 1 runoff
	if presence['soil1'] == 1:
		runoffSoilDis1 = runoff(climate['precip'][i],env['CN1'],env['soilA1'], V[11])*f[25]
	else:
		runoffSoilDis1 = 0
	
	# % Dissolved Soil Water 2 runoff
	if presence['soil2'] == 1:
		runoffSoilDis2 = runoff(climate['precip'][i],env['CN2'],env['soilA2'],V[13])*f[26]
	else:
		runoffSoilDis2 = 0
	
	# % Dissolved Soil Water 3 runoff
	if presence['soil3'] == 1:
		runoffSoilDis3 = runoff(climate['precip'][i],env['CN3'],env['soilA3'],V[15])*f[27]
	else:
		runoffSoilDis3 = 0
	
	# % Dissolved Soil Water 4 runoff
	if presence['soil4'] == 1:
		runoffSoilDis4 = runoff(climate['precip'][i],env['CN4'],env['soilA4'],V[17])*f[28]
	else:
		runoffSoilDis4 = 0

	# deep soil 1
	if presence['soil1'] == 1:
		leachSoil1 = horiFlow(k_infil1, V[29]) * f[29]
	else:
		leachSoil1 = 0

	# deep soil 2
	if presence['soil2'] == 1:
		leachSoil2 = horiFlow(k_infil2, V[30]) * f[30]
	else:
		leachSoil2 = 0

	# deep soil 3
	if presence['soil3'] == 1:
		leachSoil3 = horiFlow(k_infil3, V[31]) * f[31]
	else:
		leachSoil3 = 0

	# deep soil 4
	if presence['soil4'] == 1:
		leachSoil4 = horiFlow(k_infil4, V[32]) * f[32]
	else:
		leachSoil4 = 0


	# %% Air compartment
	# % entering air =  release vector data with independent values + advective transfer in from global environment
	AirR=release['air'][i]+airAdvection(climate['windspeed'][i],env['area'],env['airH'],env['airV'],bgConc['gairc'],ENM['density'])
	
	# %  Air f(0)
	# % loss from air is dry deposition, wet deposition,  attachment to aerosols,
	# % and air advection
	AirLoss = -(dryDepositionAir + wetDepositionAir + heteroaggregationAirAer + advectionAir)
	# % transfers into air
	SW2Air = 0 #% seawater to air is zero unless no aerosols compartment
	S2Air1 = 0 #% soil to air transfers are 0 unless there is no aerosols compartment
	S2Air2 = 0
	S2Air3 = 0
	S2Air4 = 0
	# % without aerosols compartment transfers to aersols shift to air
	if presence['aer']==0:
		# % resuspension of aerosols goes to air instead
		SW2Air = aerosolizationSW
		# % wind resuspension of soils goes to air if no aerosols
		S2Air1 = windErosionSoil1
		S2Air2 = windErosionSoil2
		S2Air3 = windErosionSoil3
		S2Air4 = windErosionSoil4
	# % without air compartment no transfer to air
	if presence['air']==0:
		SW2Air=0
		S2Air1=0
		S2Air2=0
		S2Air3=0
		S2Air4=0


	# %% Aerosols compartment
	# % entering aerosols = direct release + advective transfer in
	# bgConc['gaerc_n'] is assumed to be zero
	AerR=airAdvection(climate['windspeed'][i],env['area'],env['airH'],env['airV'],0,ENM['density'])
	# % Aerosols f(1)
	# % attachment to aerosols
	A2Aer = heteroaggregationAirAer
	
	# % loss from aerosols includes wet and dry deposition and advection
	AerLoss = -(dryDepositionAer + wetDepositionAer + advectionAer)
	
	# % transfer from seawater to aerosols by wind resuspension
	SW2Aer = aerosolizationSW #% if no aerosols this goes straight into air
	# % transfer to aerosols by wind erosion from soil
	S2Aer1 = windErosionSoil1 #% if no aerosols needs to be rerouted to air
	S2Aer2 = windErosionSoil2
	S2Aer3 = windErosionSoil3
	S2Aer4 = windErosionSoil4
	if presence['aer']==0:
		A2Aer=0
		SW2Aer=0
		S2Aer1=0
		S2Aer2=0
		S2Aer3=0
		S2Aer4=0

# %% River water compartment
	# % entering river water = direct release vector
	RwR=release['rw'][i]

	# % river water f(2)
	# % transfer from air to freshwater via wet and dry deposition
	A2RW = dryDepositionAir*np.true_divide(env['rwA'],env['area']) + wetDepositionAir*np.true_divide(env['rwA'],env['area'])
	# % loss from river water includes sedimentation, dissolution,
	# % heteroaggregation, and advective flow
	if presence['rw'] == 1:
		RWLoss = -(sedimentationRW + heteroaggregationRW + advectionRW + dissolutionRW[1])
	else:
		RWLoss = -(sedimentationRW + heteroaggregationRW + advectionRW)
	# % runoff from soil water to river water
	Sw2RW1 = runoffSoil1 #% if no river rerouted to marine, if no marine just a loss
	Sw2RW2 = runoffSoil2
	Sw2RW3 = runoffSoil3
	Sw2RW4 = runoffSoil4
	# leaching from deep soil to river water
	deepS2RW1 = leachSoil1
	deepS2RW2 = leachSoil2
	deepS2RW3 = leachSoil3
	deepS2RW4 = leachSoil4

	# % transfers to river water compartment when no river water suspended sediment
	# % compartment
	Sed2RW = 0 #%  this is zero unless there is no river water suspended sediment
	S2RW1_noSS = 0 #% this is zero unless there is no river water suspended sediment compartment
	S2RW2_noSS = 0
	S2RW3_noSS = 0
	S2RW4_noSS = 0
	# % if there is no rss compartment
	if presence['rSS']==0:
		# % resuspension from sediment to river water
		Sed2RW=resuspensionRWSed
		# % erosion from soil to river water
		S2RW1_noSS=solidErosionSoil1
		S2RW2_noSS=solidErosionSoil2
		S2RW3_noSS=solidErosionSoil3
		S2RW4_noSS=solidErosionSoil4
	if presence['rw']==0:
		A2RW=0
		Sw2RW1=0
		Sw2RW2=0
		Sw2RW3=0
		Sw2RW4=0
		Sed2RW=0
		S2RW1_noSS=0
		S2RW2_noSS=0
		S2RW3_noSS=0
		S2RW4_noSS=0
		deepS2RW1 = 0
		deepS2RW2 = 0
		deepS2RW3 = 0
		deepS2RW4 = 0


	# %% River water Suspended Sediment  f(3)
	RwSSR = release['rSS'][i]
	# % deposition from aerosols to river water suspended sediment
	Aer2RW = dryDepositionAer*np.true_divide(env['rwA'],env['area']) + wetDepositionAer*np.true_divide(env['rwA'],env['area'])
	# % transfer to suspended sediment via heteroaggregation
	RW2SS = heteroaggregationRW
	# % loss from river water suspended sediment via deposition and advective flow
	RWSSLoss = -(sedimentationRWSS + advectionRWSS)
	
	# % if no river water suspended sediment, these are rerouted to river water
	# % resuspension from sediment
	Sed2RWSS = resuspensionRWSed
	# % erosion from soils
	S2RW1 = solidErosionSoil1 #% if no river water SS rerouted to river water, if no river water, reroute to marine ss, if no marine ss reroute to marine
	S2RW2 = solidErosionSoil2
	S2RW3 = solidErosionSoil3
	S2RW4 = solidErosionSoil4
	if presence['rSS']==0:
		Aer2RW=0
		RW2SS=0
		Sed2RWSS=0
		S2RW1=0
		S2RW2=0
		S2RW3=0
		S2RW4=0


	# %% River water Sediment
	RwSedR = release['rwSed'][i]
	# % RWSediment f(4)
	# % sedimentation of free ENMs
	RW2Sed = sedimentationRW
	# % sedimentation of suspended sediment
	RWSS2Sed = sedimentationRWSS
	# % loss from sediment by resuspension, burial, dissolution, and advective
	# % transfer
	if presence['rw'] == 1:
		RWSedLoss = -(resuspensionRWSed + burialRWSed + advectionRWSed + dissolutionRWSed[1])
	else:
		RWSedLoss = -(resuspensionRWSed + burialRWSed + advectionRWSed)
	# % If there is no sediment compartment, no transfers
	if presence['rSed']==0:
		RW2Sed=0
		RWSS2Sed=0
		RWSedLoss=0

	# %% Freshwater compartment
	# % entering freshwater = direct release vector
	FwR=release['fw'][i]

	# % freshwater f(5)
	# % transfer from air to freshwater via wet and dry deposition
	A2FW = dryDepositionAir*np.true_divide(env['freshwA'],env['area']) + wetDepositionAir*np.true_divide(env['freshwA'],env['area'])
	# % loss from freshwater includes sedimentation, dissolution,
	# % heteroaggregation, and advective flow
	if presence['fw'] == 1:
		FWLoss = -(sedimentationFW + heteroaggregationFW + advectionFW + dissolutionFW[1])
	else:
		FWLoss = -(sedimentationFW + heteroaggregationFW + advectionFW)
	# % runoff from soil water to freshwater
	Sw2FW1 = runoffSoil1 #% if no freshwater rerouted to marine, if no marine just a loss
	Sw2FW2 = runoffSoil2
	Sw2FW3 = runoffSoil3
	Sw2FW4 = runoffSoil4
	# leaching from deep soil to freshwater
	deepS2FW1 = leachSoil1
	deepS2FW2 = leachSoil2
	deepS2FW3 = leachSoil3
	deepS2FW4 = leachSoil4

	# % transfers to freshwater compartment when no freshwater suspended sediment
	# % compartment
	Sed2FW = 0 #%  this is zero unless there is no freshwater suspended sediment
	S2FW1_noSS = 0 #% this is zero unless there is no freshwater suspended sediment compartment
	S2FW2_noSS = 0
	S2FW3_noSS = 0
	S2FW4_noSS = 0
	# % if there is no fss compartment
	if presence['fSS']==0:
		# % resuspension from sediment to freshwater
		Sed2FW=resuspensionFWSed
		# % erosion from soil to freshwater
		S2FW1_noSS=solidErosionSoil1
		S2FW2_noSS=solidErosionSoil2
		S2FW3_noSS=solidErosionSoil3
		S2FW4_noSS=solidErosionSoil4
	if presence['fw']==0:
		A2FW=0
		Sw2FW1=0
		Sw2FW2=0
		Sw2FW3=0
		Sw2FW4=0
		Sed2FW=0
		S2FW1_noSS=0
		S2FW2_noSS=0
		S2FW3_noSS=0
		S2FW4_noSS=0
		deepS2FW1 = 0
		deepS2FW2 = 0
		deepS2FW3 = 0
		deepS2FW4 = 0


	# %% Freshwater Suspended Sediment  f(6)
	FwSSR = release['fSS'][i]
	# % deposition from aerosols to freshwater suspended sediment
	Aer2FW = dryDepositionAer*np.true_divide(env['freshwA'],env['area']) + wetDepositionAer*np.true_divide(env['freshwA'],env['area'])
	# % transfer to suspended sediment via heteroaggregation
	FW2SS = heteroaggregationFW
	# % loss from freshwater suspended sediment via deposition and advective flow
	FWSSLoss = -(sedimentationFWSS + advectionFWSS)
	
	# % if no freshwater suspended sediment, these are rerouted to freshwater
	# % resuspension from sediment
	Sed2FWSS = resuspensionFWSed
	# % erosion from soils
	S2FW1 = solidErosionSoil1 #% if no freshwater SS rerouted to freshwater, if no freshwater, reroute to marine ss, if no marine ss reroute to marine
	S2FW2 = solidErosionSoil2
	S2FW3 = solidErosionSoil3
	S2FW4 = solidErosionSoil4
	if presence['fSS']==0:
		Aer2FW=0
		FW2SS=0
		Sed2FWSS=0
		S2FW1=0
		S2FW2=0
		S2FW3=0
		S2FW4=0


	# %% Freshwater Sediment
	FwSedR = release['fwSed'][i]
	# % FWSediment f(7)
	# % sedimentation of free ENMs
	FW2Sed = sedimentationFW
	# % sedimentation of suspended sediment
	FWSS2Sed = sedimentationFWSS
	# % loss from sediment by resuspension, burial, dissolution, and advective
	# % transfer
	if presence['fw'] == 1:
		FWSedLoss = -(resuspensionFWSed + burialFWSed + advectionFWSed + dissolutionFWSed[1])
	else:
		FWSedLoss = -(resuspensionFWSed + burialFWSed + advectionFWSed)
	# % If there is no sediment compartment, no transfers
	if presence['fSed']==0:
		FW2Sed=0
		FWSS2Sed=0
		FWSedLoss=0


	# %% Seawater
	# % direct release vector to seawater
	SwR=release['sw'][i]

	# % Seawater f(8)
	# % wet and dry deposition from air
	A2SW=dryDepositionAir*np.true_divide(env['seawA'],env['area']) + wetDepositionAir*np.true_divide(env['seawA'],env['area'])
	# % advective transfer from freshwater flow
	FW2SW=advectionFW
	# % loss processes from seawater include sedimentation, dissolution,
	# % heteroaggregation, aerosolization, and advective flow out
	if presence['sw'] == 1:
		SWLoss = -(sedimentationSW + heteroaggregationSW + aerosolizationSW + advectionSW + dissolutionSW[1])
	else: 
		SWLoss = -(sedimentationSW + heteroaggregationSW + aerosolizationSW + advectionSW)
	# % resuspension to seawater from sediment
	Sed2SW = 0 #% sed to seawater is 0 unless there is no seawater suspended sediment compartment
	if presence['sSS']==0:
		Sed2SW = resuspensionSWSed
	# % transfer form soil water to seawater
	Sw2SW1 = 0 #% soil water to seawater is equal to 0 unless freshwater compartment doesn't exist
	Sw2SW2 = 0
	Sw2SW3 = 0
	Sw2SW4 = 0
	if presence['fw']==0:
		Sw2SW1 = runoffSoil1
		Sw2SW2 = runoffSoil2
		Sw2SW3 = runoffSoil3
		Sw2SW4 = runoffSoil4
	# % transfer from soil to marine erosion
	S2SW1_noSS = 0 #% soil to seawater is 0 unless there is no freshwater ss, no freshwater, no marine ss
	S2SW2_noSS = 0
	S2SW3_noSS = 0
	S2SW4_noSS = 0
	if presence['fSS']==0 and presence['fw']==0 and presence['sSS']==0:
		S2SW1_noSS=solidErosionSoil1
		S2SW2_noSS=solidErosionSoil2
		S2SW3_noSS=solidErosionSoil3
		S2SW4_noSS=solidErosionSoil4
	# % if there is no seawater, no transfers can occur
	if presence['sw']==0:
		A2SW=0
		FW2SW=0
		Sed2SW=0
		Sw2SW1=0
		Sw2SW2=0
		Sw2SW3=0
		Sw2SW4=0
		S2SW1_noSS=0
		S2SW2_noSS=0
		S2SW3_noSS=0
		S2SW4_noSS=0

	# %% Seawater Suspended Sediment f(9)
	SwSSR = release['sSS'][i]
	# % wet and dry deposition from aerosols
	Aer2SW = dryDepositionAer*np.true_divide(env['seawA'],env['area']) + wetDepositionAer*np.true_divide(env['seawA'],env['area'])
	# % advective transfer from freshwater suspended sediment seawater suspended
	# % sediment
	FWSS2SWSS = advectionFWSS
	# % heteroaggregation with suspended sediment
	SW2SS = heteroaggregationSW
	# % losses from sw suspended sediment include sedimentation and advective
	# % flow
	SWSSLoss = -(sedimentationSWSS + advectionSWSS)
	# print SWSSLoss
	# % resuspension of sediment
	Sed2SWSS = resuspensionSWSed # % sed to seawater suspended sediment rerouted to seawater if no suspended sediment compartment
	# % soil to seawater suspended sediment is 0 unless there is no freshwater suspended sediment and no freshwater compartments
	S2SWSS1 = 0
	S2SWSS2 = 0
	S2SWSS3 = 0
	S2SWSS4 = 0
	if presence['fSS']==0 and presence['fw']==0:
		# % if no freshwater SS rerouted to freshwater, if no freshwater, reroute to marine ss, if no marine ss reroute to marine
		S2SWSS1=solidErosionSoil1
		S2SWSS2=solidErosionSoil2
		S2SWSS3=solidErosionSoil3
		S2SWSS4=solidErosionSoil4
	# % if no suspended sediment, no transfers
	if presence['sSS']==0:
		Aer2SW=0
		FWSS2SWSS=0
		SW2SS=0
		Sed2SWSS=0
		S2SWSS1=0
		S2SWSS2=0
		S2SWSS3=0
		S2SWSS4=0


	# %% Seawater Sediment
	SwSedR = release['swSed'][i]

	# % SWSediment f(10)
	# % Sedimentation of free nanoparticles
	SW2Sed = sedimentationSW
	# % sedimentation of suspended sediment
	SWSS2Sed = sedimentationSWSS
	# % advective transfer from freshwater sediment
	FWSed2SWsed = advectionFWSed
	# % loss from sediment include resuspension, burial, advection and
	# % dissolution
	if presence['sw'] == 1:
		SWSedLoss = -(resuspensionSWSed + burialSWSed + advectionSWSed + dissolutionSWSed[1])
	else: 
		SWSedLoss = -(resuspensionSWSed + burialSWSed + advectionSWSed)
	# % if there is no sediment, no transfers occur
	if presence['sSed']==0:
		SW2Sed=0
		SWSS2Sed=0
		SWSedLoss=0
		FWSed2SWsed=0


	# %% Soil Solids 1
	# % direct release vector
	S1R=release['soil1'][i]

	# % SoilSolid1 f(11)
	# % dry deposition from air
	A2Soil1 = dryDepositionAir*np.true_divide(env['soilA1'],env['area'])
	# % dry deposition from aerosols
	Aer2Soil1 = dryDepositionAer*np.true_divide(env['soilA1'],env['area'])
	# % loss from soil includes wind erosion, soil erosion, and partitioning to
	# % soil water
	SLoss1 = -(windErosionSoil1 + solidErosionSoil1 + soil2soilwater1)
	# % gain by paritioning from soil water
	SW2S1 = soilwater2soil1
	# % if there is no soil, there are no transfers
	if presence['soil1']==0:
		A2Soil1=0
		Aer2Soil1=0
		SW2S1=0


	# %% Soil Water 1 f(12)
	# % wet deposition from air
	A2SW1 = wetDepositionAir*np.true_divide(env['soilA1'],env['area'])
	# % wet deposition from aerosols
	Aer2SW1 = wetDepositionAer*np.true_divide(env['soilA1'],env['area'])
	# % transfer from soil solids partitioning to soil water
	S2SW1 = soil2soilwater1
	# % losses from runoff, dissolution, transfer to deep soil, partitoning to
	# % soil solids
	if presence['soil1'] == 1:
		SWLoss1 = -(runoffSoil1 + infiltraSoil1 + dissolutionSoil1[1] + soilwater2soil1)
	else:
		SWLoss1 = -(runoffSoil1 + infiltraSoil1 + soilwater2soil1)
	# % if no soil water, no transfers
	if presence['soilW1']==0:
		A2SW1=0
		Aer2SW1=0
		S2SW1=0


	# %% Soil Solids 2
	# % direct release vector
	S2R=release['soil2'][i]

	# % SoilSolid2 f(13)
	# % dry deposition from air
	A2Soil2 = dryDepositionAir*np.true_divide(env['soilA2'],env['area'])
	# % dry deposition from aerosols
	Aer2Soil2 = dryDepositionAer*np.true_divide(env['soilA2'],env['area'])
	# % loss from soil includes wind erosion, soil erosion, and partitioning to
	# % soil water
	SLoss2 = -(windErosionSoil2 + solidErosionSoil2 + soil2soilwater2)
	# % gain by paritioning from soil water
	SW2S2 = soilwater2soil2
	# % if there is no soil, there are no transfers
	if presence['soil2']==0:
		A2Soil2=0
		Aer2Soil2=0
		SW2S2=0


	# %% Soil Water 2 f(14)
	# % wet deposition from air
	A2SW2 = wetDepositionAir*np.true_divide(env['soilA2'],env['area'])
	# % wet deposition from aerosols
	Aer2SW2 = wetDepositionAer*np.true_divide(env['soilA2'],env['area'])
	# % transfer from soil solids partitioning to soil water
	S2SW2 = soil2soilwater2
	# % losses from runoff, dissolution, transfer to deep soil, partitoning to
	# % soil solids
	if presence['soil2'] == 1:
		SWLoss2 = -(runoffSoil2 + infiltraSoil2 + dissolutionSoil2[1] + soilwater2soil2)
	else:
		SWLoss2 = -(runoffSoil2 + infiltraSoil2 + soilwater2soil2)
	# % if no soil water, no transfers
	if presence['soilW2']==0:
		A2SW2=0
		Aer2SW2=0
		S2SW2=0   
		

	# %% Soil Solids 3
	# % direct release vector
	S3R=release['soil3'][i]

	# % SoilSolid3 f(15)
	# % dry deposition from air
	A2Soil3 = dryDepositionAir*np.true_divide(env['soilA3'],env['area'])
	# % dry deposition from aerosols
	Aer2Soil3 = dryDepositionAer*np.true_divide(env['soilA3'],env['area'])
	# % loss from soil includes wind erosion, soil erosion, and partitioning to
	# % soil water
	SLoss3 = -(windErosionSoil3 + solidErosionSoil3 + soil2soilwater3)
	# % gain by paritioning from soil water
	SW2S3 = soilwater2soil3
	# % if there is no soil, there are no transfers
	if presence['soil3']==0:
		A2Soil3=0
		Aer2Soil3=0
		SW2S3=0


	# %% Soil Water 3 f(16)
	# % wet deposition from air
	A2SW3 = wetDepositionAir*np.true_divide(env['soilA3'],env['area'])
	# % wet deposition from aerosols
	Aer2SW3 = wetDepositionAer*np.true_divide(env['soilA3'],env['area'])
	# % transfer from soil solids partitioning to soil water
	S2SW3 = soil2soilwater3
	# % losses from runoff, dissolution, transfer to deep soil, partitoning to
	# % soil solids
	if presence['soil3'] == 1:
		SWLoss3 = -(runoffSoil3 + infiltraSoil3 + dissolutionSoil3[1] + soilwater2soil3)
	else:
		SWLoss3 = -(runoffSoil3 + infiltraSoil3 + soilwater2soil3)
	# % if no soil water, no transfers
	if presence['soilW3']==0:
		A2SW3=0
		Aer2SW3=0
		S2SW3=0   


	# %% Soil Solids 4
	# % direct release vector
	S4R=release['soil4'][i]

	# % SoilSolid4 f(17)
	# % dry deposition from air
	A2Soil4 = dryDepositionAir*np.true_divide(env['soilA4'],env['area'])
	# % dry deposition from aerosols
	Aer2Soil4 = dryDepositionAer*np.true_divide(env['soilA4'],env['area'])
	# % loss from soil includes wind erosion, soil erosion, and partitioning to
	# % soil water
	SLoss4 = -(windErosionSoil4 + solidErosionSoil4 + soil2soilwater4)
	# % gain by paritioning from soil water
	SW2S4 = soilwater2soil4
	# % if there is no soil, there are no transfers
	if presence['soil4']==0:
		A2Soil4=0
		Aer2Soil4=0
		SW2S4=0


	# %% Soil Water 4 f(18)
	# % wet deposition from air
	A2SW4 = wetDepositionAir*np.true_divide(env['soilA4'],env['area'])
	# % wet deposition from aerosols
	Aer2SW4 = wetDepositionAer*np.true_divide(env['soilA4'],env['area'])
	# % transfer from soil solids partitioning to soil water
	S2SW4 = soil2soilwater4
	# % losses from runoff, dissolution, transfer to deep soil, partitoning to
	# % soil solids
	if presence['soil4'] == 1:
		SWLoss4 = -(runoffSoil4 + infiltraSoil4 + dissolutionSoil4[1] + soilwater2soil4)
	else:
		SWLoss4 = -(runoffSoil4 + infiltraSoil4 + soilwater2soil4)	# % if no soil water, no transfers
	if presence['soilW4']==0:
		A2SW4=0
		Aer2SW4=0
		S2SW4=0
	
	# %% River water Dissolved f19
	# % Loss from dissolved advection
	RWDisLoss = -(advectionRWDis)
	# % River water dissolved
	if presence['rw'] == 1:
		RW2Dis = dissolutionRW[0]
	else:
		RW2Dis = 0
	# % runoff to river water
	Soil2RWDis = runoffSoilDis1 + runoffSoilDis2 + runoffSoilDis3 + runoffSoilDis4

	
	# %% River water Sediment Dissolved f20
	# % Loss from dissolved advection
	RWSedDisLoss = -(advectionRWSedDis)
	# % Freshwater sediment dissolved 
	if presence['rw'] == 1:
		RWSed2Dis = dissolutionRWSed[0]
	else:
		RWSed2Dis = 0
		
	# %% Freshwater Dissolved f21
	# % Loss from dissolved advection
	FWDisLoss = -(advectionFWDis)
	# % Freshwater dissolved
	if presence['fw'] == 1:
		FW2Dis = dissolutionFW[0]
	else:
		FW2Dis = 0
	# % runoff to freshwater
	Soil2FWDis = runoffSoilDis1 + runoffSoilDis2 + runoffSoilDis3 + runoffSoilDis4

	
	# %% Freshwater Sediment Dissolved f22
	# % Loss from dissolved advection
	FWSedDisLoss = -(advectionFWSedDis)
	# % Freshwater sediment dissolved 
	if presence['fw'] == 1:
		FWSed2Dis = dissolutionFWSed[0]
	else:
		FWSed2Dis = 0
		
	# %% Marine Dissolved f23
	# % loss from dissolved advection
	SWDisLoss = -(advectionSWDis)
	# % marine dissolved
	if presence['sw'] == 1:
		sw2Dis = dissolutionSW[0]
		# % freshwater to marine dissolved advection
		FW2SWDis = advectionFWDis
	else:
		sw2Dis = 0
		FW2SWDis = 0

	
	# %% Marine Sediment Dissolved f24
	# % loss from dissolved advection
	SWSedDisLoss = -(advectionSWSedDis)
	# % marine dissolved
	if presence['sw'] == 1:
		SWSed2Dis = dissolutionSWSed[0]
		# % freshwater sediment to marine sediment
		FWSed2SWSedDis = advectionFWSedDis
	else:
		SWSed2Dis = 0
		FWSed2SWSedDis = 0


	# %% Soil Water 1 Dissolved f25
	# % loss from runoff
	SW1DisLoss = -(runoffSoilDis1)
	# % dissolution
	if presence['soil1'] ==1:
		SW1Dis = dissolutionSoil1[0]
	else:
		SW1Dis = 0
	
	# %% Soil Water 2 Dissolved f26
	# % loss from runoff
	SW2DisLoss = -(runoffSoilDis2)
	# % dissolution
	if presence['soil2'] == 1:
		SW2Dis = dissolutionSoil2[0]
	else:
		SW2Dis = 0
	
	# %% Soil Water 3 Dissolved f27
	# % loss from runoff
	SW3DisLoss = -(runoffSoilDis3)
	# % dissolution
	if presence['soil3'] == 1:
		SW3Dis = dissolutionSoil3[0]
	else:
		SW3Dis = 0
	
	# %% Soil Water 4 Dissolved f28
	# % loss from runoff
	SW4DisLoss = -(runoffSoilDis4)
	# % dissolution
	if presence['soil4'] == 1:
		SW4Dis = dissolutionSoil4[0]
	else:
		SW4Dis = 0

		
	#transferValues=np.array([dryDepositionAir,wetDepositionAir,heteroaggregationAirAer,advectionAir,dryDepositionAer,
	#	wetDepositionAer,advectionAer,sedimentationFW,heteroaggregationFW,advectionFW,dissolutionFW,sedimentationFWSS,
	#	advectionFWSS,resuspensionFWSed,burialFWSed,advectionFWSed,dissolutionFWSed,sedimentationSW,heteroaggregationSW,
	#	aerosolizationSW,advectionSW,dissolutionSW,sedimentationSWSS,advectionSWSS,resuspensionSWSed,burialSWSed,
	#	advectionSWSed,dissolutionSWSed,windErosionSoil1,solidErosionSoil1,soil2soilwater1,runoffSoil1,infiltraSoil1,
	#	soilwater2soil1,dissolutionSoil1,windErosionSoil2,solidErosionSoil2,soil2soilwater2,runoffSoil2,infiltraSoil2,
	#	soilwater2soil2,dissolutionSoil2,windErosionSoil3,solidErosionSoil3,soil2soilwater3,runoffSoil3,infiltraSoil3,
	#	soilwater2soil3,dissolutionSoil3,windErosionSoil4,solidErosionSoil4,soil2soilwater4,runoffSoil4,infiltraSoil4,
	#	soilwater2soil4,dissolutionSoil4])
	
	#with open('/Users/kgarner/Documents/Python/Python/Nano_Model/Solve_py/transferValues.csv','ab') as f:
	#	writer=csv.writer(f)
	#	writer.writerow(transferValues)


	# %% total results
	results=[
		AirR + AirLoss + SW2Air + S2Air1 + S2Air2 + S2Air3 + S2Air4,
		AerR + A2Aer + AerLoss + SW2Aer + S2Aer1 + S2Aer2 + S2Aer3 + S2Aer4,
		RwR + A2RW + RWLoss + Sw2RW1 + Sw2RW2 + Sw2RW3 + Sw2RW4 + Sed2RW + S2RW1_noSS + S2RW2_noSS + S2RW3_noSS + S2RW4_noSS + deepS2RW1 + deepS2RW2 + deepS2RW3 + deepS2RW4,
		RwSSR + Aer2RW + RW2SS + RWSSLoss + Sed2RWSS + S2RW1 + S2RW2 + S2RW3 + S2RW4,
		RwSedR + RW2Sed + RWSS2Sed + RWSedLoss,
		FwR + A2FW + FWLoss + Sw2FW1 + Sw2FW2 + Sw2FW3 + Sw2FW4 + Sed2FW + S2FW1_noSS + S2FW2_noSS + S2FW3_noSS + S2FW4_noSS + deepS2FW1 + deepS2FW2 + deepS2FW3 + deepS2FW4,
		FwSSR + Aer2FW + FW2SS + FWSSLoss + Sed2FWSS + S2FW1 + S2FW2 + S2FW3 + S2FW4,
		FwSedR + FW2Sed + FWSS2Sed + FWSedLoss,
		SwR + A2SW + FW2SW + SWLoss + Sed2SW + Sw2SW1 + Sw2SW2 + Sw2SW3 + Sw2SW4 + S2SW1_noSS + S2SW2_noSS + S2SW3_noSS + S2SW4_noSS,
		SwSSR + Aer2SW + FWSS2SWSS + SW2SS + SWSSLoss + Sed2SWSS + S2SWSS1 + S2SWSS2 + S2SWSS3 + S2SWSS4,
		SwSedR + FWSed2SWsed + SW2Sed + SWSS2Sed + SWSedLoss,
		S1R + A2Soil1 + Aer2Soil1 + SLoss1 + SW2S1,
		A2SW1 + Aer2SW1 + S2SW1 + SWLoss1,
		S2R + A2Soil2 + Aer2Soil2 + SLoss2 + SW2S2,
		A2SW2 + Aer2SW2 + S2SW2 + SWLoss2,
		S3R + A2Soil3 + Aer2Soil3 + SLoss3 + SW2S3,
		A2SW3 + Aer2SW3 + S2SW3 + SWLoss3,
		S4R + A2Soil4 + Aer2Soil4 + SLoss4 + SW2S4,
		A2SW4 + Aer2SW4 + S2SW4 + SWLoss4,
		RW2Dis + RWDisLoss + Soil2RWDis,
		RWSed2Dis + RWSedDisLoss,
		FW2Dis + FWDisLoss + Soil2FWDis,
		FWSed2Dis + FWSedDisLoss,
		sw2Dis + SWDisLoss + FW2SWDis,
		SWSed2Dis + SWSedDisLoss + FWSed2SWSedDis,
		SW1Dis + SW1DisLoss,
		SW2Dis + SW2DisLoss,
		SW3Dis + SW3DisLoss,
		SW4Dis + SW4DisLoss,
		infiltraSoil1 - leachSoil1,
		infiltraSoil2 - leachSoil2,
		infiltraSoil3 - leachSoil3,
		infiltraSoil4 - leachSoil4
	]

	
	return results
	


	