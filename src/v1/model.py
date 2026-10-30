import os
from pathlib import Path

import joblib
from src.v1.preprocessing import preprocessor, load_data
from src.v1.explain import explain_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyarrow

ver = 1

#EXTRA CODE FOR EXPORTING DATASETS AND CHECKING CORRELATION
def export_processed_dataset(ver=1):

    X_train, X_test, y_train, y_test = load_data(ver=ver)

    #preprocessing x_train and x_test
    x_train_processed, x_train_processed_std = preprocessor(X_train, fit_transform='yes', std='yes', ver=ver)
    x_test_processed, x_test_processed_std = preprocessor(X_test, fit_transform='no', std='yes', ver=ver)

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    
    #exporting them
    os.makedirs(f"data/processed/v{ver}", exist_ok=True)
    
    x_train_processed.to_parquet(f"data/processed/v{ver}/x_train_processed.parquet", index=False)
    x_train_processed_std.to_parquet(f"data/processed/v{ver}/x_train_processed_std.parquet", index=False)
    x_test_processed.to_parquet(f"data/processed/v{ver}/x_test_processed.parquet", index=False)
    x_test_processed_std.to_parquet(f"data/processed/v{ver}/x_test_processed_std.parquet", index=False)
    y_train.to_parquet(f"data/processed/v{ver}/y_train.parquet", index=False)
    y_test.to_parquet(f"data/processed/v{ver}/y_test.parquet", index=False)

    print(f'x_train_processed {x_train_processed.shape}')
    print('x_test_processed ',x_test_processed.shape)
    print('y_train ', y_train.shape)
    print('y_test ', y_test.shape)
    
    #for sanity check
    # print(x_train_processed.info())
    # print(x_train_processed_std.info())
    #print(x_test_processed.info())
    #print(x_test_processed_std.info())
    print(x_train_processed.describe())
    #print(x_train_processed_std.describe())
    print(x_test_processed.describe())
    #print(x_test_processed_std.describe())

    ax = sns.heatmap(
        x_train_processed.corr(), 
        annot=True, 
        square=True, 
        fmt='.2f',
        linewidths=1, 
        linecolor='gray',
        annot_kws={"size": 8}
    )

    plt.xticks(
        rotation=45, 
        ha='right', 
        rotation_mode='anchor',
        fontsize=10
    )

    plt.yticks(fontsize=10)

    plt.tight_layout()

    os.makedirs(f"plots/features/", exist_ok=True)
    
    plt.savefig(f"plots/features/feature_correlation.png", bbox_inches="tight", dpi=300)
    plt.show()


#EXTRA CODE FOR EXPLAINING THE MODEL
def model_explain():
    ver = 1
    _, X_test, _, _ = load_data(ver=ver)
    
    x_test_processed = preprocessor(X_test, fit_transform='no', std='no', ver=ver)
    

    name = "XGBClassifier"
    model = joblib.load(f'models\\models\\v1\\best_model_overall_1.pkl')
    
    explainer = "tree"
    
    generated_images = explain_model(
        x_test=x_test_processed.head(100),
        model=model,
        name=name,
        explainer_type=explainer,
        ver=ver,
        target_class=1
    )

    output_dir = Path(f"plots/shap/v{ver}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(generated_images):
        image_path = output_dir / f"{name}_shap_plot_{idx + 1}.png"
        img.save(image_path)
        print(f"Saved plot: {image_path}")
        
model_explain()