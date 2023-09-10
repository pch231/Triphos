#!/usr/bin/env python
import os
import numpy as np
import subprocess
from pbi import ligandtools
from pbi.pdbtools import PDBtools
from openbabel import pybel
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue
import argparse


def docking(vina, config_file, ligand_file, output_file):
    run_line = f"{vina} --config {config_file} --ligand {ligand_file} --out {output_file}"
    try:
        subprocess.check_output(run_line.split(), stderr=subprocess.STDOUT, universal_newlines=True)
    except Exception as e:
        return e
    return None


def read_score(dock_pdb_file, vina_v='vina'):
    fp = open(dock_pdb_file)
    lines = fp.readlines()
    fp.close()
    score_list = list()
    for line in lines:
        if vina_v == 'vina':
            if line.startswith('REMARK VINA RESULT'):
                score = float(line[19:29].strip())
                score_list.append(score)
        elif vina_v == 'smina':
            if line.startswith('REMARK minimizedAffinity'):
                score = float(line[24:36].strip())
                score_list.append(score)
    score_list = np.array(score_list)
    return score_list


def creator(q, data, num_sub_proc):
    for idx, d in enumerate(data):
        q.put((idx, d))
    for _ in range(num_sub_proc):
        q.put('DONE')


def worker(q, param, return_dict):
    vina, config_file, dock_dir, vina_v = param
    while True:
        qqq = q.get()
        if qqq == 'DONE':
            break

        (idx, pdb_file) = qqq

        mol_id = os.path.splitext(os.path.basename(pdb_file))[0]
        mol_id2 = mol_id[:7]
        dock_dir1 = os.path.join(dock_dir, mol_id2)
        if not os.path.exists(dock_dir1):
            try:
                os.makedirs(dock_dir1)
            except FileExistsError as e:
                print(e, flush=True)

        pdbqt_file = os.path.join(dock_dir1, f"{mol_id}.pdbqt")
        dock_pdbqt_file = os.path.join(dock_dir1, f"dock_{mol_id}.pdbqt")
        dock_pdb_file = os.path.join(dock_dir1, f"dock_{mol_id}.pdb")
        pdb_file_full = os.path.join("ligands", pdb_file)  # Full path to PDB file

        PDBtools.ligand_to_pdbqt(pdb_file_full, pdbqt_file)
        docking(vina, config_file, pdbqt_file, dock_pdbqt_file)
        ligandtools.pdbqt_to_pdb_ref(dock_pdbqt_file, dock_pdb_file, pdb_file_full)  # Use full path here
        dock_score = read_score(dock_pdb_file, vina_v=vina_v)

        return_dict[idx] = dock_score


def main():

    parser = argparse.ArgumentParser(description='docking with multi process')
    parser.add_argument('-v', '--vina', type=str, required=True,
                        default='vina', help='vina run file ')
    parser.add_argument('-c', '--dock_config', type=str, required=True,
                        default='config.txt', help='docking config file ')
    parser.add_argument('-l', '--pdb_list', type=str, required=True,
                        default='pdb_list.txt', help='list of PDB files')
    parser.add_argument('-d', '--out_dir', type=str, required=True,
                        default='dock', help='output directory')
    parser.add_argument('-o', '--score_file', type=str, required=False,
                        default='docking.txt', help='score file')
    parser.add_argument('-p', '--ncpu', type=int, required=False,
                        default=1, help='number of subprocesses')

    args = parser.parse_args()

    vina = args.vina
    config_file = args.dock_config
    pdb_list_file = os.path.join("ligands", args.pdb_list)  # 경로 수정
    out_dir = args.out_dir
    score_file = args.score_file
    num_sub_proc = args.ncpu

    vina_v = 'vina'
    if vina == 'smina':  # vina or smina
        vina_v = 'smina'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(pdb_list_file) as fp:
        pdb_files = [line.strip() for line in fp.readlines()]

    num_data = len(pdb_files)
    num_sub_proc = min(num_sub_proc, num_data)

    param = (vina, config_file, out_dir, vina_v)

    q1 = Queue()
    manager = Manager()
    return_dict = manager.dict()
    proc_master = Process(target=creator, args=(q1, pdb_files, num_sub_proc))
    proc_master.start()

    procs = []
    for _ in range(num_sub_proc):
        proc = Process(target=worker, args=(q1, param, return_dict))
        procs.append(proc)
        proc.start()
    q1.close()
    q1.join_thread()
    proc_master.join()
    for proc in procs:
        proc.join()

    with open(score_file, 'w') as fp:
        line_out = 'mol_id docking_score\n'
        fp.write(line_out)
        for idx, pdb_file in enumerate(pdb_files):
            mol_id = os.path.splitext(os.path.basename(pdb_file))[0]
            line_out = f'{mol_id}'
            if idx in return_dict:
                dscore = return_dict[idx]
                line_score = ' '.join([f'{score:.3f}' for score in dscore])
                line_out = f'{line_out} {line_score}'
            else:
                line_out += ' None'
            line_out += '\n'
            fp.write(line_out)


if __name__ == "__main__":
    main()
    
    