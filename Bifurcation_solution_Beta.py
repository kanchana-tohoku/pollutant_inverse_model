#This code will target to perform apportionment
# not only at a single bifurcation, but to a for a river system
#======================================================

#importing necessary python libraries
    #PCRaster for handling maps
    #Numpy for NNLS analysis
    #

# User Input data
    #composotion and location of the observed pollutant signal
        #concentration [mg/m3]: eg:- {Cd, Cr, Pd, Cu, Al}
        #location [Lat/ Long]
        #time of observation:- {date, time}
     
#Load initial data available:
    #pre defined flow paths
    #map of industries and pollutant signatures of them
        #Location IDs and pollutant signatures
    #Flow data (discharge/velocity) until the  time of pollutant detection
    #confluence cells map
        # list of confluence cells [raw, column] {location: x,y}
            #create confluence map
        
#creating a class to NNLS solving     

#outputs of the model
    #location of pollutant discharge source (or the ID of the source)
        


