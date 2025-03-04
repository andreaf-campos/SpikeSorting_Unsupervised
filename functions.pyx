#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Barreras - Campos

MÓDULO DE FUNCIONES PARA LA GRAFICACIÓN Y EL CÁLCULO DE CIERTOS PROCESOS DE LA CLASIFICACIÓN DE ESPIGAS.

"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from math import ceil
from seaborn import heatmap
from scipy.stats import pearsonr, norm
from pandas import DataFrame

#### ------------------------------- FUNCIONES PARA GRÁFICAS --------------------------------------####
    
#Funcion para obtener la matriz de densidad para la gráfica de calor de las Features, así como los ejes de la gráfica:
def axis_heatmap(Spike_Features):
    """
    Esta funcion calcula la discretizacion de puntos de las características de las espigas, para poder formar
    las graficas de calor. También calcula sus ejes.
    Para esto requiere:
    
    1.- Spike_Features: arreglo en el que vienen las Características Extraídad y Seleccionadas de las Espigas.
    """
    
    #This section will calculate densities between axis 1,3 and 1,2
    minf1  = min(Spike_Features[:, 0])
    maxf1  = max(Spike_Features[:, 0])
    stepf1 = (maxf1 - minf1)/50
    minf2  = min(Spike_Features[:, 1])
    maxf2  = max(Spike_Features[:, 1])
    stepf2 = (maxf2 - minf2)/50
    minf3  = min(Spike_Features[:, 2])
    maxf3  = max(Spike_Features[:, 2])
    stepf3 = (maxf3 - minf3)/50
    pc1    = np.arange(minf1,maxf1+stepf1,stepf1) 
    pc2    = np.arange(maxf2,minf2-stepf2,-stepf2)
    pc3    = np.arange(maxf3,minf3-stepf3,-stepf3)
    x1     = len(pc1)
    x2     = len(pc2)
    x3     = len(pc3)
    
    densidad  = np.zeros((x1, x2))
    densidad2 = np.zeros((x1, x3))
    densidad3 = np.zeros((x1, x3))
    
    for x1i in range(x1-1):
        for x2i in range(x2-1):
        
            cond1 = np.where((Spike_Features[:,0] >= pc1[x1i]) & (Spike_Features[:,0] < pc1[x1i+1]))[0]
            cond2 = np.where((Spike_Features[:,1] > pc2[x2i+1]) & (Spike_Features[:,1] <= pc2[x2i]))[0]
            inter = np.intersect1d(cond1, cond2)
            densidad[x2i][x1i] =  len(inter)
            
        for x3i in range(x3-1):
            cond1 = np.where((Spike_Features[:,1] >= pc2[-1-x1i+1]) & (Spike_Features[:,1] < pc2[-1-x1i]))[0]
            cond2 = np.where((Spike_Features[:,2] > pc3[x3i+1]) & (Spike_Features[:,2] <= pc3[x3i]))[0]
            inter = np.intersect1d(cond1, cond2)
            densidad2[x3i][x1i] = len(inter)
        
        for x3i in range(x3-1):
            cond1= np.where((Spike_Features[:,0]>=pc1[x1i]) & (Spike_Features[:,0]<pc1[x1i+1]))[0]
            cond2= np.where((Spike_Features[:,2]>pc3[x3i+1]) & (Spike_Features[:,2]<=pc3[x3i]))[0]
            inter= np.intersect1d(cond1, cond2)
            densidad3[x3i][x1i]= len(inter)

    #Definición de los ejes para los mapas de calor        
    Feature1 = []
    Feature2 = []
    Feature3 = []
    for i in range(len(pc1)):
        if i%int(len(pc1)/10) == 0:
            Feature1.append(round(pc1[i],4))
            Feature2.append(round(pc2[i],4))
            Feature3.append(round(pc3[i],4))
        else:
            Feature1.append("")
            Feature2.append("")
            Feature3.append("")  
            
    return densidad, densidad2, densidad3, Feature1, Feature2, Feature3
    


#Función para graficar la forma de las espigas pertenecientes a cada cluster y la proyeccion en 3D de las características con
#su color de cluster correspondiente:
def show_classification(labels, features, waveforms, fase,img_path, save= False):
    """Esta función ilustra los resultados de la clasificación de las formas de onda, para ello necesita de  
       entradas, de las cuales labels es obligatoria y puede estar cualquiera de las otras tres. La
       descripción de las entradas se lista a continuación:
       
       1.- labels (entrada obligada), es un vector que indica las clases por posición de cada una de los
       datos categorizados.
       
       2.- feautures:   (Opcional) Es una matriz, los renglones indican cada una de las observaciones muestrales mientras que las columnas indican la dimensión de cada observación.
       
       3.- waveforms: (Opcional) Se trata de una matriz, los renglón es una forma de onda muestral.
       
       4.- fase: (Opcional) Se trata de una celda con dos elementos. El primero es una matriz donde cada renglón representa la primera derivada e la forma de onda correspondiente a la observación muestral en cada uno de los otros argumentos de entrada. El segundo elemento de la celda representa la segunda derivada.
       
       5.- img_path: La dirección donde queremos que se guarde la imagen.
       
       6.- save: True o False. Si es True, las graficas se guardaran en img_path
       
       """
    
    #Filtro: fase debe ser de tamaño 2 o 0
    assert len(fase)==2 or len(fase)==0, "Error en los parámetros de entrada, vea documentación"    
    
    colores= [(1, 0, 0), (0, 0, 1), (0, 1, 0), np.array([83,134,139])/255, (1, 1, 0), np.array([238,118,33])/255, (1,0,1), (0,0,0), np.array([255,193,193])/255, (0, 1, 1)]
    colores2= [ "r", "b", "g", "y", "m", "gray", "pink", "cyan" ]  

    nc=np.unique(labels)
    if nc[0] == -1:
        nc= nc+2    
        labels = labels+2
        colores=[(0,0,0),(1, 0, 0), (0,1, 0), (0,0,1), np.array([83,134,139])/255, (1, 1, 0), np.array([238,118,33])/255, (1,0,1), np.array([255,193,193])/255, (0, 1, 1)]
        colores2=["k", "r", "b", "g", "y", "m", "gray", "pink", "cyan" ]
     
    else:
        nc= nc+2
        labels= labels+2 
    
    #Obtención del vector de color para poder graficar en terminos del cluster al que pertenece cada elemento
    colorvec = np.zeros((len(labels),3))
    for i in range(len(colorvec)):
        for j in range(3):
            colorvec[i,j] = colores[labels[i]][j] 
                      
    #Raster tridimensional de los distintos clusters
    if np.size(features, 1)>=3:
        IMG = str('%s/Clusters3D.png' %img_path)
        plt.figure(figsize = (10, 8))
        ax = plt.axes(projection ="3d")
        ax.scatter3D(features[:,0], features[:,1], features[:,2], c = colorvec,alpha = 0.7)
        ax.set_xlabel("\nX", size=12)
        ax.set_ylabel("\nY", size=12)
        ax.set_zlabel("\nZ", size=12)
        #plt.show() 
    if save == True: 
        plt.savefig(IMG)

    #Gráfica de las distintas espigas según el cluster al que pertenecen
    IMG = str('%s/Spike_classif.png' %img_path)
    if np.size(waveforms, 0)>0:
        if max(nc)>6:
            fig, axs = plt.subplots(3,ceil(len(nc)/2), figsize=(15, 6))
            axs = axs.ravel()
            for i in range(len(nc)):
                axs[i].plot(waveforms[labels==nc[i]].T,color=colores2[i])
            if save == True: 
                plt.savefig(IMG)
        else:
            fig, axs = plt.subplots(2,ceil(len(nc)/2), figsize=(15, 6))
            axs = axs.ravel()
            for i in range(len(nc)):
                axs[i].plot(waveforms[labels==nc[i],:].T,color=colores2[i])
            if save == True:
                plt.savefig(IMG)
                
                
#Función para graficar los raster plots de las neuronas clasificadas:
def rasterplot(spike_times,spike_trial,clases, img_path, *args, save = False):
    
    """Realiza el raster de los resultados del Spike Sorting. Para ello requiere los siguientes argumentos de
       entrada obligatorios
       
       1.- Spike_Times: vector con los tiempos de los potenciales de acción
       
       2.- Spike_Trials: vector de ensayos con la informacion del ensayo al cual pertenece el vector de espigas
       
       3.- Clases: Vector o matriz con la clasificación de cada espiga. 
       
       4.- img_path: La dirección donde se quiere guardar la imagen
       
       5.- Save: True o False. Indica si deseas que se guarden los rasters. 
       
       La clasificación -1 indica ruido. De este modo, el comando: Rasterplot(Spike_Times, Spike_Trials, clases) genera tantos rasters como clases haya,  sin ningún tipo de alineación artificial.
       """

    #spike_times, spike_trial y Spike_class son 2D, por lo que debemos convertirlos en vectores 1D
    
    if len(args) == 1:                    #En caso de que si se meta el input "Spike_Class" 
        amp = np.zeros(len(args[0]))      #Almacena su contenido en la variable "amp"
        Spike_Times = np.zeros(len(amp))
        Spike_Trial = np.zeros(len(amp))
        for i in range(len(amp)):
            amp[i] = args[0][i]
            Spike_Times[i] = spike_times[i]
            Spike_Trial[i] = spike_trial[i]
    elif len(args) == 0:
        amp=np.ones(len(spike_times))    #En caso de que no se introduzca el argumento "Spike_Class"
                                         #Definimos a la variable "amp" como un vector de con tantos como elementos en Spike_Times
        Spike_Times = np.zeros(len(amp))
        Spike_Trial = np.zeros(len(amp))
        for i in range(len(amp)):
            Spike_Times[i] = spike_times[i]
            Spike_Trial[i] = spike_trial[i]
    
    elem = clases.ndim         #Dimensionalidad del vector clases (en matlab es solo una columna, en python es solo
                               #una fila)
    
    if np.max(Spike_Times)>10000:
        Spike_Times=Spike_Times/30000 #Reescalamos el tiempo a segundos 
    elif np.max(Spike_Times)>1000:
        Spike_Times=Spike_Times/1000  #Reescalamos el tiempo a segundos 
            
     
    #Generación de máscaras para la selección rápida de datos.
    
    if elem>1:                #Es una clasificación suave. Las columnas indican número de clusters
        
        raise NameError("Incorrect format of classification array, 1D array")   
    
    elif elem==1:             #Es una clasificación dura. el máximo indica #clusters
        
        clusters= np.unique(clases)
    
        if clusters[0]==-1:         #Si tenemos elementos clasificados como ruido, generamos una mascara con estos
            masknoise= clases==-1
            clusters = clusters[1:]
        else:
            masknoise= np.zeros(len(clases)) #Si no tenemos ruido, la mascara estará llena de ceros
    
    else:
        raise NameError("Incorrect format of classification array, empty array")

    namps= np.unique(amp)
    for nras in range(len(clusters)):
        plt.figure(figsize=(12,7))
        plt.title(str('Cluster %d' %(nras+1)))
        y=0
        for metadata in range(len(namps)):
            mask=   np.logical_and(clases==nras,amp==namps[metadata])
            trials= np.unique(Spike_Trial[mask])
            for tr in range(len(trials)):
                mask2= Spike_Trial==trials[tr]
                plt.scatter(Spike_Times[mask & mask2], np.ones(sum(mask & mask2))*y,color='black', marker= '|')
                y+=0.1
                
        if save == True:
            IMG = str('%s/Raster_neu%d.png' %(img_path,nras+1))
            plt.savefig(IMG)

    #Ahora grafica el raster de lo que se consideró ruido 
    plt.figure(figsize=(12,7))
    plt.title('Noise')
    y=0
    for metadata in range(len(namps)):
        mask =  np.logical_and(masknoise,amp==metadata)
        trials= np.unique(Spike_Trial[mask])
        for tr in range(len(trials)):
            plt.scatter(Spike_Times[mask],np.ones(sum(mask))*y,color = "black", marker='|')
            y+=0.1
    if save == True:
        IMG = str('%s/Raster_noise.png' %(img_path))
        plt.savefig(IMG)

        
#Función para hacer el mapa de calor de la forma de las espigas de las neuronas clasificadas:
def spikesheat_plot(Spike_Waveforms_denoised_aligned, Nc, rmnoise, kmeans_labels, img_path, save= False):
    """
    Realiza el gráfico de calor de las formas de espigas de las neuronas clasificadas. Para esto requiere:
    1.- Spike_Waveforms_denoised_aligned: Arreglo de todas las formas de onda sin ruido y alineadas.
    
    2.- Nc: int, número de clusters
    
    3.- rmnoise: int, indica si se hizo la remocion de ruido de los clusteres clasificados. Si es 1 es que sí se hizo la remoción 
    
    4.- kmeans_labels: vector que indica las clases por posición de cada una de los datos categorizados.
    
    5.- img_path: dirección donde se guardaran las graficas.
    
    6.- save: True o False. Si es True guarda la grafica de estabilidad.
    """
    
    #Graficas de calor de espigas 
    waveform_grid= np.linspace(np.max(Spike_Waveforms_denoised_aligned)+8, np.min(Spike_Waveforms_denoised_aligned)-8,40)

    if  (Nc+1)>6:                            #En caso de que la cantidad de clusters que buscamos sea 6 o mayor
                                             #tendremos (3,ceil((Nc+1)/3) subgráficas
        nrowfig,ncolfig= 3,ceil((Nc+1)/3)

    else:                                    #En caso de que busquemos menos clusters, tendremos (3,ceil((Nc+1)/3)
                                             #subgráficas
        nrowfig,ncolfig= 2, ceil((Nc+1)/2)     


    #Definición del eje_y para los mapas de calor        
    amplitude = []
    for i in range(len(waveform_grid)):
        if i%int(len(waveform_grid)/7) == 0:              #Tomamos 7 elementos con igual espaciamiento entre ellos
            amplitude.append(round(waveform_grid[i],4))
        else:
            amplitude.append("")                          #Los elementos no tomados, los sustituimos por un str vacío


    if rmnoise != 1:                         #En caso de haber decidido no hacer denoising

        IMG = str('%s/Spike_HeatMap.png' %img_path)
        fig,axs = plt.subplots(nrowfig,ncolfig,figsize=(25,15))
        axs = axs.ravel()
        for Nclus in range(Nc):
            tmp=       kmeans_labels ==Nclus
            x4=        len(waveform_grid)
            x5=        np.arange(0,np.size(Spike_Waveforms_denoised_aligned, 1),1), #Aquí no se si empezar desde 0 o desde 1
            densidad3= np.zeros((x4,len(x5[0])))

            for x1i in range(x4):
                for x2i in range(len(x5[0])):
                    densidad3[x1i, x2i]= sum(((Spike_Waveforms_denoised_aligned[tmp, x5[0][x2i]])>(waveform_grid[x1i]-4)) & ((Spike_Waveforms_denoised_aligned[tmp, x5[0][x2i]])<(waveform_grid[x1i]+4)))

            g = heatmap(densidad3,  vmin = 0, vmax= np.max(densidad3)*0.35, xticklabels= x5[0], yticklabels= amplitude, cmap= "hot", ax=axs[Nclus])        
            g.set_ylabel('Amplitude')
            nameplot = str('Neuron %d' %(Nclus+1))
            g.set_title(nameplot)

        if save == True:
            plt.savefig(IMG)

    elif rmnoise==1:                         #En caso de haber decidido hacer denoising

        fracc = 0.35                         #Fracción que determina la escala de calor de la gráfica
        flag = 1
        while flag:
            fig,axs = plt.subplots(nrowfig,ncolfig,figsize=(25,15))
            axs = axs.ravel()  
            for Nclus in range(Nc+1):
                if Nclus<Nc:
                    tmp=       kmeans_labels ==Nclus
                    x4=        len(waveform_grid)
                    x5=        np.arange(0,np.size(Spike_Waveforms_denoised_aligned,1),1) #Aquí no se si empezar desde 0 o desde 1
                    densidad3= np.zeros((x4,len(x5)))

                    for x1i in range(x4):
                        for x2i in range(len(x5)):
                            densidad3[x1i, x2i]= sum(((Spike_Waveforms_denoised_aligned[tmp, x5[x2i]])>(waveform_grid[x1i]-4)) & ((Spike_Waveforms_denoised_aligned[tmp, x5[x2i]])<(waveform_grid[x1i]+4)))

                    g = heatmap(densidad3,  vmin = 0, vmax= np.max(densidad3)*fracc, xticklabels= x5, yticklabels= amplitude, cmap= "hot", ax=axs[Nclus])        
                    g.set_ylabel('Amplitude')
                    nameplot = str('Neuron %d' %(Nclus+1))
                    g.set_title(nameplot)


                else:
                    tmp=       kmeans_labels ==-1
                    x4=        len(waveform_grid)
                    x5=        np.arange(0,np.size(Spike_Waveforms_denoised_aligned, 1),1)
                    densidad3= np.zeros((x4,len(x5)))
                    for x1i in range(x4):
                        for x2i in range(len(x5)):
                            densidad3[x1i, x2i]= sum(((Spike_Waveforms_denoised_aligned[tmp, x5[x2i]])>(waveform_grid[x1i]-4)) & ((Spike_Waveforms_denoised_aligned[tmp, x5[x2i]])<(waveform_grid[x1i]+4)))

                    if 0 < np.max(densidad3):
                        g = heatmap(densidad3,  vmin = 0, vmax= np.max(densidad3)*fracc, xticklabels= x5, yticklabels= amplitude, cmap= "hot", ax=axs[Nclus])        
                        g.set_ylabel('Amplitude')
                        g.set_title("Noise")
                        IMG = str('%s/Spikes_HeatMap%.2f.png' %(img_path,fracc))
                        fig = plt.gcf()
                        plt.show()
                        
                        flag = int(input('¿Quieres cambiar la fracción de la gráfica de ruido? \n1.SI\n0.NO \n '))
                        if flag == 1:
                            fracc = float(input('Introduce el nuevo valor de la fracción (0,1]:  '))
                        else:
                            if save== True:
                                fig.savefig(IMG)
                            else:
                                break

                    else:
                        print('\nNo se detectó ruido')
                        IMG = str('%s/Spikes_HeatMap%.2f.png' %(img_path,fracc))
                        if save == True:
                            plt.savefig(IMG)
                        flag = 0
                        
                        
        
#Función para la gráfica de estabilidad de la actividad de las neuronas clasificadas por ensayo:
def stability_plot(kmeans_labels, Spike_Trial, img_path, save= False):
    """
    Realiza el gráfico de estabilidad de las neuronas clasificadas. Para esto requiere:
    
    1.- kmeans_labels: vector que indica las clases por posición de cada una de los datos categorizados.
    
    2.- Spike_Trials: vector de ensayos con la informacion del ensayo al cual pertenece el vector de espigas
    
    3.- img_path: dirección donde se guardaran las graficas.
    
    4.- save: True o False. Si es True guarda la grafica de estabilidad.
    """
    cl= np.unique(kmeans_labels)
    trials = np.unique(Spike_Trial)
    nsppertrial= np.zeros((len(trials), len(cl)), dtype= int)

    if min(cl) == -1:
        colors= ["k", "r", "b", "g", "y", "m", "gray", "pink", "cyan" ]
    else:
        colors= ["r", "b", "g", "y", "m", "gray", "pink", "cyan" ]

    labels= []
    for i in range(len(cl)):
        if cl[i]== -1:
            labels.append("Noise")
        else:
            labels.append(str("Neuron %d" %(cl[i]+1)))   

    plt.figure(figsize=(10,6))
    for i in range(len(cl)):
        mask = kmeans_labels== cl[i]
        for tr in range(len(trials)):
            masktr = Spike_Trial == trials[tr]
            nsppertrial[tr, i]= sum(masktr.ravel()*mask)

        plt.plot(trials, nsppertrial[:,i], colors[i], label= labels[i]) #trials, 

    plt.legend(loc="best")
    plt.xlabel("Number of trial", size=12)
   # plt.show()
    
    if save == True:
        plt.savefig(str('%s/Stability.png' %img_path))
         
    return nsppertrial, labels

        
    
    
    
###---------------------------------- FUNCIONES PARA CÁLCULOS -----------------------------------###

#Clase de matplotlib.mlab que calcula el Analisis de Componentes Principales:
#El codigo fuente lo conseguí de https://matplotlib.org/3.0.2/_modules/matplotlib/mlab.html#PCA
class PCA(object):
    def __init__(self, a, standardize=True):
        """
        compute the SVD of a and store data for PCA.  Use project to
        project the data onto a reduced set of dimensions

        Parameters
        ----------
        a : np.ndarray
            A numobservations x numdims array
        standardize : bool
            True if input data are to be standardized. If False, only centering
            will be carried out.

        Attributes
        ----------
        a
            A centered unit sigma version of input ``a``.

        numrows, numcols
            The dimensions of ``a``.

        mu
            A numdims array of means of ``a``. This is the vector that points
            to the origin of PCA space.

        sigma
            A numdims array of standard deviation of ``a``.

        fracs
            The proportion of variance of each of the principal components.

        s
            The actual eigenvalues of the decomposition.

        Wt
            The weight vector for projecting a numdims point or array into
            PCA space.

        Y
            A projected into PCA space.

        Notes
        -----
        The factor loadings are in the ``Wt`` factor, i.e., the factor loadings
        for the first principal component are given by ``Wt[0]``. This row is
        also the first eigenvector.

        """
        n, m = a.shape
        if n < m:
            raise RuntimeError('we assume data in a is organized with '
                               'numrows>numcols')

        self.numrows, self.numcols = n, m
        self.mu = a.mean(axis=0)
        self.sigma = a.std(axis=0)
        self.standardize = standardize

        a = self.center(a)

        self.a = a

        U, s, Vh = np.linalg.svd(a, full_matrices=False)

        # Note: .H indicates the conjugate transposed / Hermitian.

        # The SVD is commonly written as a = U s V.H.
        # If U is a unitary matrix, it means that it satisfies U.H = inv(U).

        # The rows of Vh are the eigenvectors of a.H a.
        # The columns of U are the eigenvectors of a a.H.
        # For row i in Vh and column i in U, the corresponding eigenvalue is
        # s[i]**2.

        self.Wt = Vh

        # save the transposed coordinates
        Y = np.dot(Vh, a.T).T
        self.Y = Y

        # save the eigenvalues
        self.s = s**2

        # and now the contribution of the individual components
        vars = self.s / len(s)
        self.fracs = vars/vars.sum()

    def project(self, x, minfrac=0.):
        '''
        project x onto the principle axes, dropping any axes where fraction
        of variance<minfrac
        '''
        x = np.asarray(x)
        if x.shape[-1] != self.numcols:
            raise ValueError('Expected an array with dims[-1]==%d' %
                             self.numcols)
        Y = np.dot(self.Wt, self.center(x).T).T
        mask = self.fracs >= minfrac
        if x.ndim == 2:
            Yreduced = Y[:, mask]
        else:
            Yreduced = Y[mask]
        return Yreduced


    def center(self, x):
        '''
        center and optionally standardize the data using the mean and sigma
        from training set a
        '''
        if self.standardize:
            return (x - self.mu)/self.sigma
        else:
            return (x - self.mu)


    @staticmethod
    def _get_colinear():
        c0 = np.array([
            0.19294738,  0.6202667,   0.45962655,  0.07608613,  0.135818,
            0.83580842,  0.07218851,  0.48318321,  0.84472463,  0.18348462,
            0.81585306,  0.96923926,  0.12835919,  0.35075355,  0.15807861,
            0.837437,    0.10824303,  0.1723387,   0.43926494,  0.83705486])

        c1 = np.array([
            -1.17705601, -0.513883,   -0.26614584,  0.88067144,  1.00474954,
            -1.1616545,   0.0266109,   0.38227157,  1.80489433,  0.21472396,
            -1.41920399, -2.08158544, -0.10559009,  1.68999268,  0.34847107,
            -0.4685737,   1.23980423, -0.14638744, -0.35907697,  0.22442616])

        c2 = c0 + 2*c1
        c3 = -3*c0 + 4*c1
        a = np.array([c3, c0, c1, c2]).T
        return a
    
    
#Algoritmo K-means para la formacion de clusters (Se usará esta funcion en el programa solo para el caso de Nclus= 1):
def kmeans(datos, ncentros, niter = 100, tolerancia = np.exp(-3), centroides = None):
    """
    Función de Andrea Campos del algoritmo de clustering K-means.
    La función cálcula la clasificación de una matriz de datos en N-clusters indicados por el usuario mediante el argumento ncentros.

    Parameters
    ----------
    datos : TYPE array
        DESCRIPTION. La matriz de datos que se desea clasificar en clusters.
    ncentros : TYPE int
        DESCRIPTION. El número de clusters en los que agruparemos a nuestros datos. Es el número de centroides.
    niter : TYPE, optional. int
        DESCRIPTION. El default es 100. El número de iteraciones que realizará el algoritmo para actualizar los centroides y la clasificación.
                    NOTA: Puede que no se lleguen a cumplir todas las iteraciones ya que el algoritmo se detiene cuando convergen los centroides.
    tolerancia : TYPE, optional. float o int
        DESCRIPTION. El default es np.exp(-3). El umbral de tolerancia de la distancia entre los centroides de la iteración actual y sus predecesores.
                    Le indica al algoritmo cuando parar las actualizaciones de los centroides y la clasificación al llegar a la convergencia entre centroides.
    centroides : TYPE, optional. array
        DESCRIPTION. El default es None. Cuando es None entonces el algoritmo iniciará con centroides aleatorios. 
                    Cambia cuando ingresamos el arreglo de centroides iniciales de manera manual. 
                    RECUERDA: El arreglo de centroides iniciales debe ser una MATRIZ  de forma (n-centros X n-columnas de la matriz de datos).

    Returns
    -------
    clasificador: TYPE array
        DESCRIPTION. El vector que contiene la clasificación de los datos a uno de los centroides. Tendrá tantos elementos como filas tenga la matriz de datos,
                    y cada elemento corresponde al número del centroide al que pertenece esa fila de datos. 
    centroides: TYPE array
        DESCRIPTION. Arreglo de los n-centroides de la última iteracion (actualización). Su forma será de (n-centros x n-columnas de la matriz de datos)
    iteraciones (x+1): TYPE int
        DESCRIPTION. Número de iteraciones que realizó el algoritmo hasta la convergencia de los centroides.

    """
    clasificador= np.zeros(len(datos), dtype= int)  #Arreglo donde se actualizaran las clasificaciones de los datos a los centroides
    
    if centroides is None:
        Centroides = []   #Lista donde se irán guardando todos los ncentroides con sus n-componentes de cada iteracion
        
        for x in range(niter):        
            while x == 0:
                #1. En la primer iteracion elegimos npuntos aleatorios dentro del rango (min:max): 
                puntos= np.random.uniform(np.min(datos), np.max(datos), ncentros)
                
                #2. Cálculo de la distancia euclideana entre los puntos aleatorios y los datos en cada renglon de la matriz de datos
                dist= np.zeros((len(datos), ncentros))   #Matriz donde se guardaran las distancias
                for i in range(len(datos)):
                    for j in range(ncentros):
                        dist[i,j]= np.sqrt(np.sum((datos[i,:]-puntos[j])**2)) #Cálculo de la distancia euclideana.
                
                #3. Clasificación de los datos al punto más cercano
                for i in range(len(clasificador)):
                    dist_min = np.where(dist[i,:] == np.min(dist[i,:]))[0] #Encontramos los índices donde esta la menor(es) distancia(s)
                
                    if len(dist_min) == 1:  #Si hay un solo punto con la menor distancia, entonces:
                        clasificador[i] = int(dist_min + 1) #Se guarda en el vector el índice "natural" del punto (por eso el +1)
                        
                    else: #Si hay más de un punto que esta cerca (mas de un punto con la misma distancia euclideana), entonces:
                        clasificador[i] =  int(np.random.choice(dist_min) +1) #Se elige aleatoriamente uno de esos puntos y se guarda en el vector el índice "natural" del punto elegido (por eso el +1)
                
                #Evaluamos si todos los puntos iniciales (centroides) están dentro del clasificador para poder salir del while y avanzar con las siguientes iteraciones.
                if np.all(np.isin(np.linspace(1, ncentros, ncentros), clasificador)):  #Si todos los puntos sí están en el clasificador, entonces salimos del while
                    x+=1
                else:   #Si todos los puntos NO estan en el clasificador, entonces nos quedamos en el while ya que el iterador se queda en 0.
                    x = 0
                        
            else:
                #1. Se actualizan los n centroides usando el clasificador resultante de la iteracion x-1
                cents = []  #Lista temporal donde se guardan los n-centroides de la iteracion actual
                for i in range(ncentros):
                    idx = np.where(clasificador== i + 1)[0] 
                    centroide = np.mean(datos[idx,::], axis=0)
                    cents.append(centroide)
                Centroides.append(cents)  #Agregamos los centroides actualizados de la iter. actual en la lista de centroides general
                    
                #2.Distancias Euclideanas entre los n centroides y los renglones de la matriz de datos
                distancias= np.zeros((len(datos), ncentros))  #Matriz donde se guardan las distancias eucl.
                
                for i in range(len(datos)):
                    for j in range(ncentros):
                        distancias[i,j]= np.sqrt(np.sum((datos[i,:]-cents[j])**2))  #Distancia euclid. entre centroide y los datos[i, :]
                
                #3.Clasificacion de puntos a un centroide            
                for i in range(len(datos)):
                    dist_min = np.where(distancias[i,:] == np.min(distancias[i,:]))[0] #Encontramos los índices donde esta la menor(es) distancia(s)
                
                    if len(dist_min) == 1:  #Si hay un solo punto con la menor distancia, entonces:
                        clasificador[i] = int(dist_min + 1) #Se guarda en el vector el índice "natural" del punto (por eso el +1)
                        
                    else: #Si hay más de un punto que esta cerca (mas de un punto con la misma distancia euclideana), entonces:
                        clasificador[i] = int(np.random.choice(dist_min) +1) #Se elige aleatoriamente uno de esos puntos y se guarda en el vector el índice "natural" del punto elegido (por eso el +1)
                        
                #4. Evaluación de la convergencia de los centroides
                if x>1:
                    dist_centroides = np.zeros(ncentros)
                    for i in range(ncentros):
                        dist_centroides[i] = np.sqrt(np.sum((Centroides[x][i]- Centroides[x-1][i])**2))
                        
                    cent_tol = np.where(dist_centroides <= tolerancia)[0]
                    if len(cent_tol)== ncentros:
                        break
                    else:
                        continue
        return clasificador, Centroides[-1], x+1

    else:
        assert(len(centroides) == ncentros), "El valor del argumento ncentros debe ser igual al número de centroides ingresados, es decir, el número de filas de la matriz de centroides debe ser igual al valor de ncentros"
        assert(len(centroides[0,:])== len(datos[0, :])), "El número de componentes (elementos) de cada centroide debe ser igual al número de columnas de la matriz de datos."
        Centroides = np.zeros((np.shape(centroides)))  
        
        for x in range(niter):
            
            if x == 0:
                
                #1. Cálculo de la distancia euclideana entre los centroides ingresados y los datos en cada renglon de la matriz de datos
                dist= np.zeros((len(datos), ncentros))   #Matriz donde se guardaran las distancias
                for i in range(len(datos)):
                    for j in range(ncentros):
                        dist[i,j]= np.sqrt(np.sum((datos[i,:]-centroides[j,:])**2)) #Cálculo de la distancia euclideana.
                
                #2. Clasificación de los datos al centroide más cercano
                for i in range(len(clasificador)):
                    dist_min = np.where(dist[i,:] == np.min(dist[i,:]))[0] #Encontramos los índices donde esta la menor(es) distancia(s)
                
                    if len(dist_min) == 1:  #Si hay un solo centroide con la menor distancia, entonces:
                        clasificador[i] = int(dist_min + 1) #Se guarda en el vector el índice "natural" del centroide (por eso el +1)
                        
                    else: #Si hay más de un centroide que esta cerca (mas de un centroide con la misma distancia euclideana), entonces:
                        clasificador[i] =  int(np.random.choice(dist_min) +1) #Se elige aleatoriamente uno de esos centroides y se guarda en el vector su índice "natural" (por eso el +1)
                
                #3. Evaluamos si todos los centroides iniciales son ÓPTIMOS para la clasificación, es decir, si están TODOS dentro del clasificador .
                if np.all(np.isin(np.linspace(1, ncentros, ncentros), clasificador)):  #Si todos los centroides sí están en el clasificador, entonces seguimos con las iteraciones
                    Centroides = centroides  #Guardamos los centroides ingresados en el arreglo de Centroides porque SÍ son óptimos
                    x+=1
                else:   #Si todos los centroides NO estan en el clasificador, entonces rompemos el ciclo y marcamos error al usuario
                    return print("ERROR: Los centroides ingresados NO son óptimos para la clasificación, por favor ingrese nuevos centroides.")

            else:
                
                #1. Se actualizan los n centroides usando el clasificador resultante de la iteracion x-1
                nuevos_centroides= np.zeros((np.shape(centroides)))  #Arreglo donde se guardan los centroides actualizados (de la iteracion actual)
                
                for i in range(ncentros):
                    idx = np.where(clasificador== i + 1)[0] 
                    nuevos_centroides[i,:]= np.mean(datos[idx,::], axis=0)      
                    
                #2.Distancias Euclideanas entre los n centroides y los renglones de la matriz de datos
                distancias= np.zeros((len(datos), ncentros))  #Matriz donde se guardan las distancias eucl.
                
                for i in range(len(datos)):
                    for j in range(ncentros):
                        distancias[i,j]= np.sqrt(np.sum((datos[i,:]- nuevos_centroides[j, :])**2))  #Distancia euclid. entre centroide y los datos[i, :]
                
                #3.Clasificacion de puntos a un centroide            
                for i in range(len(clasificador)):
                    dist_min = np.where(distancias[i,:] == np.min(distancias[i,:]))[0] #Encontramos los índices donde esta la menor(es) distancia(s)
                
                    if len(dist_min) == 1:  #Si hay un solo centroide con la menor distancia, entonces:
                        clasificador[i] = int(dist_min + 1) #Se guarda en el vector el índice "natural" del centroide (por eso el +1)
                        
                    else: #Si hay más de un centroide que esta cerca (mas de un centroide con la misma distancia euclideana), entonces:
                        clasificador[i] =  int(np.random.choice(dist_min) +1) #Se elige aleatoriamente uno de esos centroides y se guarda en el vector su índice "natural" (por eso el +1)
                                        
                #4. Evaluación de la convergencia de los centroides    
                dist_centroides = np.zeros(ncentros) #Vector deonde se guardará el valor de la distancia entre el centroide N actual (iteracionx) y su predecesor (x-1)
                
                for i in range(ncentros):
                    dist_centroides[i] = np.sqrt(np.sum((nuevos_centroides[i,:]- Centroides[i,:])**2))  #Nuevos_centroides son los centroides actualizados (iteracion x), mientras que Centroides son los centroides pasados (iter x-1)
                    
                cent_tol = np.where(dist_centroides <= tolerancia)[0]  #ïndices donde se cumpla que la distancia es menor a la tolerancia
                if len(cent_tol)== ncentros:  #Si todos los centroides tienen una distancia menor a la tolerancia entonces rompemos el ciclo
                    break
                else:  #Si NO todos los centroides tienen una distancia < tolerancia, entonces actualizamos el arreglo de Centroides y continuamos con el ciclo.
                        #Se actualiza el arreglo de Centroides guardando los centroides de la iteracion actual, y ya en la siguiente iteración, este arreglo corresponderá a los centroides de la iteracion x-1
                    Centroides = nuevos_centroides
                    continue
                    
        return clasificador, nuevos_centroides, x+1
    
    
#Función para calcular los coeficientes de correlación entre el número de espigas por ensayo entre neuronas. Así como el valor de p y su intervalo de confianza del 95%:
def corrcoef(matrix):
    """
    Calcula el coeficiente R de la correlacion de Pearson entre todas las combinaciones posibles de columnas en matrix, así como su valor de p que indica la probabilidad de que las columnas no esten correlacionadas, y el rango de valores R (minimo y maximo) para un intervalo de confianza del 95%.
    Lo que requiere es:
    
    1.- matrix: arreglo. Se harán los cálculos correspondientes entre cada par de columnas dentro de la matriz
    """
    
    dframe= DataFrame(matrix)

    fmatrix = dframe.values
    rows, cols = fmatrix.shape

    r = np.zeros((cols, cols), dtype=float)
    p = np.zeros((cols, cols), dtype=float)
    lo = np.zeros((cols, cols), dtype=float)
    hi = np.zeros((cols, cols), dtype=float)

    for i in range(cols):
        for j in range(cols):
            if i == j:
                r_, p_, lo_, hi_ = 1., 1., 1., 1.  #En la diagonal todos valen 1
            else:
                #Cálculo del coeficiente de correlacion de pearson (R) y su p-value
                r_, p_ = pearsonr(fmatrix[:,i], fmatrix[:,j])
                
                #Cálculo de los intervalos de confianza 95% de R
                r_z = np.arctanh(r_)  #Transformar el coeficiente de correlacion en un z-score de Fisher
                se = 1/np.sqrt(fmatrix[:,i].size-3) #Calculo del error estandar (std)
                alpha = 0.05 #umbral del intervalo de confianza, en este caso 95%
                
                #Calculo del intervalo de confianza CI mediante la formula:
                #r_z +- z alpha/2 x se
                z = norm.ppf(1-alpha/2)  #z alpha/2 
                lo_z, hi_z = r_z-z*se, r_z+z*se
                #Por ultimo, revertimos la transformacion con tanh
                lo_, hi_ = np.tanh((lo_z, hi_z))

            r[j][i] = r_
            p[j][i] = p_
            lo[j][i] = lo_
            hi[j][i] = hi_

    return r, p, lo, hi