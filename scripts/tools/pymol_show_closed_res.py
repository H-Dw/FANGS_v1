from pymol import cmd
import math

# Register the command for direct invocation from the PyMOL command line.
@cmd.extend
def select_nearest_residues(target_sel, n_neighbors=15, out_sel="nearest15", protein_obj="all"):
    """
    DESCRIPTION
        Select the N residues nearest to a target region on the basis of Cα distances,
        excluding the target residues themselves.

    USAGE
        select_nearest_residues target_sel, [n_neighbors], [out_sel], [protein_obj]

    ARGS
        target_sel  : Target selection (for example, resi 100).
        n_neighbors : Number of nearest residues to retain (default, 15).
        out_sel     : Name of the output selection (default, nearest15).
        protein_obj : Search scope; all objects by default (all).
    """
    
    # Coerce n_neighbors to an integer.
    n_neighbors = int(n_neighbors)

    # 1. Collect Cα coordinates for the target region.
    # Selection logic: target selection, restrict to CA, then retrieve the model.
    target_ca_sel = f"({target_sel}) and name CA"
    target_model = cmd.get_model(target_ca_sel)

    if len(target_model.atom) == 0:
        print(f"[Error] Target selection '{target_sel}' contains no CA atoms.")
        return

    # Extract the target coordinate list.
    target_coords = [a.coord for a in target_model.atom]

    # 2. Collect candidate atoms: all protein Cα atoms outside the target region.
    # Exclude the target with the PyMOL operator 'and not' rather than testing each atom in the loop.
    # byres excludes the entire residue, not only atoms that overlap the target selection.
    candidate_sel = f"({protein_obj}) and polymer.protein and name CA and not (byres ({target_sel}))"
    candidate_model = cmd.get_model(candidate_sel)
    
    if len(candidate_model.atom) == 0:
        print("[Warning] No candidate residues found around target.")
        return

    neighbors = []

    # 3. Compute distances.
    # Iterate over all candidate Cα atoms.
    for atom in candidate_model.atom:
        # Minimum distance from this atom to any target Cα.
        # Compatibility branch for Python versions that do not provide math.dist.
        if hasattr(math, 'dist'):
            min_dist = min(math.dist(atom.coord, t_coord) for t_coord in target_coords)
        else:
            min_dist = min(
                math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(atom.coord, t_coord)))
                for t_coord in target_coords
            )
        
        # Store (atom identifier, minimum distance, residue label for reporting).
        neighbors.append((atom.id, min_dist, f"{atom.chain}/{atom.resi}{atom.resn}"))

    # 4. Rank candidates and retain the first N.
    # Sort by ascending distance.
    neighbors.sort(key=lambda x: x[1])
    top_n = neighbors[:n_neighbors]

    if not top_n:
        print("[Info] No residues found.")
        return

    # 5. Construct the selection.
    # Atom identifiers give the most direct selection, in the form id 1+2+3...
    id_list = [str(item[0]) for item in top_n]
    id_sel_str = "+".join(id_list)
    
    # Select these Cα atoms first.
    cmd.select(out_sel, f"id {id_sel_str}")
    # Expand the selection to complete residues (byres).
    cmd.select(out_sel, f"byres {out_sel}")

    # 6. Report the selected residues.
    print(f"=================================================")
    print(f"[SUCCESS] Selected {len(top_n)} nearest residues into '{out_sel}'.")
    print(f"Target: {target_sel}")
    print(f"Closest residues:")
    for i, (atom_id, dist, res_info) in enumerate(top_n):
        print(f"  {i+1}. {res_info} \t(Dist: {dist:.2f} A)")
    print(f"=================================================")

# === Example invocation (uncomment the block below to run this file as a script) ===
# When the script is loaded in the PyMOL GUI, leave the block below commented
# and enter: select_nearest_residues target_region
'''
select_nearest_residues(
    target_sel="target_region",
    n_neighbors=15,
    out_sel="nearest15"
)
'''