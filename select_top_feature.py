
import pandas as pd


def select_top_features(combined_correlations_df, num_features):

    print('Start Select Top Features')
   
    top_features = combined_correlations_df.nlargest(num_features * 2, 'Combined Correlation')
    # print('top_features:', top_features)
    
   
    selected_features = []
    used_base_features = {}  
    
    for index, row in top_features.iterrows():
        feature_name = row['Feature']  
        combined_correlation = row['Combined Correlation']  
        # print('feature_name:', feature_name)
        
       
        if '+' in feature_name:
            base_features = feature_name.split('+')
        elif '-' in feature_name:
            base_features = feature_name.split('-')
        elif '*' in feature_name:
            base_features = feature_name.split('*')
        else:
            base_features = [feature_name]  

        base_features_set = frozenset(base_features)
        
        # 
        if base_features_set not in used_base_features:
            used_base_features[base_features_set] = (feature_name, combined_correlation)
            selected_features.append(row)
        else:
            existing_feature, existing_correlation = used_base_features[base_features_set]
            if combined_correlation > existing_correlation:
            
                used_base_features[base_features_set] = (feature_name, combined_correlation)
                selected_features = [f for f in selected_features if f['Feature'] != existing_feature]
                selected_features.append(row)

        if len(selected_features) == num_features:
            break

    top_features_selected_df = pd.DataFrame(selected_features)
    #print('top_features_selected:', top_features_selected_df)
    print('End Select Top Features')
    
    top_feature_indices = top_features_selected_df.index
    return top_feature_indices

