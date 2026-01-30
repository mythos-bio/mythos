Test data in this directory generated from template directory

  data/templates/martini/m2/DMPC/273K/

Setting the number of steps to 100000 and nstxout to 1000. Then, within that
directory:

  gmx grompp -f md.mdp -c membrane.gro -p topol.top -n index.ndx -o test
  gmx mdrun -deffnm test -ntmpi 1 -rdd 1.5
  echo "LJ" | gmx energy -f test.edr -s test.tpr -o lj.xvg

the outputs test.trr, test.tpr and lj.xvg (energies for the LJ potential) were
copied to this directory.