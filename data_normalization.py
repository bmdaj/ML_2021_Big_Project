import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

def transform_parameters (parameters, transformation = "quantile"):
    """
    This function is used to normalize the parameters.
    @param parameters: pd.DataFrame of parameters to normalize
    @param transformation: Type of normalization applied i.e. quantile, standard...
    """
    
    columns = parameters.columns
    
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    import pandas as pd
    
    if transformation == "quantile":
        t = QuantileTransformer()
        trans_params =  t.fit_transform(parameters)
        
    if transformation == "standard":
        t = StandardScaler()
        trans_params =  t.fit_transform(parameters)
    
    return pd.DataFrame(trans_params, columns=columns), t


def antitransform_parameters (parameters, transformer):
    """
    This function is used to normalize the parameters.
    @param parameters: pd.DataFrame of normalized parameters
    @param transformer: Transformer used for normalization
    """  
    
    columns = parameters.columns

    return pd.DataFrame(transformer.inverse_transform(parameters), columns=columns)



def normalize_bands (frequencies, num_bands = 5, num_k_points  = 31, transformation = "quantile"):
    
    """
    This function is used to normalize the bands individually.
    @ frequencies: pd.DataFrame of simulated frequencies per k-points
    @ num_bands: Number of bands in simulations
    @ num_k_points: Number of k-points in simulations
    @ transformation: Type of normalization applied i.e. quantile, standard...
    """
    
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    import pandas as pd
    
    transformers = []
    
    
    for i in range(int(num_bands)):
        
        column_start = "Band_"+str(i)+"_k_0"
        column_end = "Band_"+str(i)+"_k_"+str(num_k_points-1)
    
        df_band = frequencies.loc[:, column_start : column_end]
        columns = df_band.columns
        
        if transformation == "quantile":
        
            t = QuantileTransformer()
            trans_band =  t.fit_transform(df_band)
            trans_band = pd.DataFrame(trans_band, columns=columns)
            transformers.append(t)
        
        if transformation == "standard":
        
            t = StandardScaler()
            trans_band =  t.fit_transform(df_band) 
            trans_band = pd.DataFrame(trans_band, columns=columns)
            transformers.append(t)
            
        if i == 0:
            
            transformed_bands = trans_band.copy()
            
        else:
            
            transformed_bands = pd.concat([transformed_bands, trans_band], axis=1)
    
    return transformed_bands, transformers


def unormalize_bands (frequencies, transformers, num_bands = 5, num_k_points  = 31):
    
    """
    This function is used to normalize the bands individually.
    @ frequencies: pd.DataFrame of simulated frequencies per k-points
    @ transformers: List of transformers returned by normalizer
    @ num_bands: Number of bands in simulations
    @ num_k_points: Number of k-points in simulations
    """
    
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    import pandas as pd
    
    
    for i in range(int(num_bands)):
        
        column_start = "Band_"+str(i)+"_k_0"
        column_end = "Band_"+str(i)+"_k_"+str(num_k_points-1)
    
        df_band = frequencies.loc[:, column_start : column_end]
        columns = df_band.columns
        
        transformer = transformers [i]
        antitrans_band = pd.DataFrame(transformer.inverse_transform(df_band), columns=columns)
            
        if i == 0:
            
            antitransformed_bands = antitrans_band.copy()
            
        else:
            
            antitransformed_bands = pd.concat([antitransformed_bands, antitrans_band], axis=1)
    
    return antitransformed_bands