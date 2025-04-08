from pathlib import Path

root_dir = Path("../IXI_dataset/IXI_raw/")
seqs = ["T1", "T2", "PD"]

subjects_dict = {}

for seq_path in root_dir.iterdir():

    if not seq_path.is_dir():
        continue
    seq_name = seq_path.name.split("-")[-1]
    # print(f"Sequence: {seq_name}")

    if seq_name not in seqs:
        continue

    print(f"Number of subjects in {seq_name}: ", len(list(seq_path.iterdir())))
    subjects_dict[seq_name] = [
        i.name.split(f"{seq_name}.nii.gz")[0] for i in list(seq_path.iterdir())
    ]

# T1-T1: {'IXI500-Guys-1017-', 'IXI309-IOP-0897-', 'IXI182-Guys-0792-', 'IXI116-Guys-0739-'}
# T2-T1: {'IXI580-IOP-1157-'}
print(f"T1-T2: {set(subjects_dict['T1']) - set(subjects_dict['T2'])}")
print(f"T2-T1: {set(subjects_dict['T2']) - set(subjects_dict['T1'])}")

# Empty: T1 & PD have the same subjects
print(f"T2-PD: {set(subjects_dict['T2']) - set(subjects_dict['PD'])}")

for seq in ["T11"]:
    print(f"Number of subjects in {seq}: {len(subjects_dict[seq])}")
