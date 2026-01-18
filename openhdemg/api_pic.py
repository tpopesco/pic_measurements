from library import tools as emg
from library.openfiles import emg_from_demuse
from library.df_bh import compute_braceheight  # BH/angle/etc. function
from library.pic import compute_deltaf as compute_df_from_pic  # DeltaF from separate module
import pandas as pd
import os
from pathlib import Path
from tkinter import Tk, filedialog

def load_emg_file(file_path):
    emgfile = emg_from_demuse(filepath=file_path)
    smoothfits = emg.compute_svr(emgfile=emgfile)["gensvr"]
    return emgfile, smoothfits

def main():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Choisir un dossier de fichiers .mat")

    if not folder_path:
        print("Aucun dossier sélectionné.")
        return

    output_dir = Path(folder_path) / "Results_PICs123"
    output_dir.mkdir(exist_ok=True)

    deltaf_excel = output_dir / "deltaf_results.xlsx"
    bh_excel = output_dir / "braceheight_results.xlsx"

    all_deltaf = []
    all_bh = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".mat"):
            file_path = os.path.join(folder_path, filename)
            try:
                print(f"Analyse de {filename}...")
                emg_data, smoothfits = load_emg_file(file_path)

                # deltaF only
                deltaf_df = compute_df_from_pic(
                    emgfile=emg_data,
                    smoothfits=smoothfits,
                    average_method="all",
                    normalisation="False",
                    recruitment_difference_cutoff=1.0,
                    corr_cutoff=0.7,
                    controlunitmodulation_cutoff=0.5,
                    clean=True,
                )
                deltaf_df["Filename"] = filename
                all_deltaf.append(deltaf_df)

                # BH + related metrics
                bh_df = compute_braceheight(
                    emgfile=emg_data,
                    smoothfits=smoothfits,
                    file_path=file_path,
                    average_method="all",
                )
                bh_df["Filename"] = filename
                all_bh.append(bh_df)

            except Exception as e:
                print(f"Erreur lors du traitement de {filename} :")


    if all_deltaf:
        pd.concat(all_deltaf, ignore_index=True).to_excel(deltaf_excel, index=False)
        print(f"Fichier deltaF généré : {deltaf_excel}")

    if all_bh:
        pd.concat(all_bh, ignore_index=True).to_excel(bh_excel, index=False)
        print(f"Fichier brace height généré : {bh_excel}")

    if not all_deltaf and not all_bh:
        print("Aucun résultat à enregistrer.")

if __name__ == "__main__":
    main()