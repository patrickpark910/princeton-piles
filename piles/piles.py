from pathlib import Path
import argparse
import os

from python.model      import Pile
from python.parameters import *
from python.utilities  import *


def main():

    DIR_FIGURES.mkdir(parents=True, exist_ok=True) 
    DIR_OPENMC.mkdir(parents=True, exist_ok=True) 

    parser = argparse.ArgumentParser(description=f"Choose ONE run type with -r flag: ['tallies', 'volume', 'plot']. Choose blankets with -b flag, multiple separated by spaces: {BLANKETS}. Choose fertile isotopes with -i flag, multiple separated by spaces: {ISOTOPES}")

    parser.add_argument("-r", "--run_type", 
                        type=str, default='tallies', 
                        help="Specify run type: ['tallies', 'volume', 'plot']" )

    parser.add_argument("-b", "--blankets", 
                        type=str, nargs="+", default=BLANKETS,
                        help=f"Specify blankets, separated by space, among this list: {BLANKETS}" )

    parser.add_argument("-i", "--isotopes", 
                        type=str, nargs="+", default=ISOTOPES,
                        help=f"Specify fertile isotopes, separated by space, among this list: {ISOTOPES}" )
    
    parser.add_argument("-f", "--fertile",
                        type=float, nargs="+", default=FERTILE_KGM3,
                        help=f"Specify fertile loadings in kg/m3 (default: {FERTILE_KGM3})")

    parser.add_argument("-p", "--pile_code", 
                        type=str, default=None, 
                        help=f"Specify pile code" )

    parser.add_argument("-np", "--n_particles", 
                        type=lambda x: int(float(x)), default=N_PARTICLES,
                        help=f"Specify number of particles in integer scientific notation, e.g. 1e6" )
    
    parser.add_argument("-nc", "--n_cycles", 
                        type=lambda x: int(float(x)), default=N_CYCLES,
                        help=f"Specify number of cycles in integers, e.g. 25" )
    
    parser.add_argument("--no_xml", 
                        dest="print_xml", action="store_false",
                        help="Disable printing model.xml files")
    
    parser.add_argument("--no_run", 
                        dest="run_openmc", action="store_false",
                        help="Disable running the OpenMC calculation")

    parser.add_argument("--exit_pixel_cm", type=float, default=EXIT_PIXEL_CM,
                        help="Maximum neutron exit pixel width in cm (default: 1)")
    parser.add_argument("--exit_energy_bins", type=int, default=EXIT_ENERGY_BINS,
                        help="Number of logarithmic exit energy bins (default: 300)")

    parser.add_argument("--chain_file", type=str, default=None,
                        help="Depletion chain for a02 (default: chain_endfb81_fast.xml beside the XS directory)")
    parser.add_argument("--gamma_energy_bins", type=int, default=1000,
                        help="Number of log photon energy bins for a02")

    # Set defaults for the 'dest' targets above so they are True unless the flag is used
    parser.set_defaults(print_xml=True, run_openmc=True, run_debug=True)

    args = parser.parse_args()

    run_type    = args.run_type
    pile_code   = args.pile_code
    blankets    = args.blankets
    isotopes    = args.isotopes
    densities   = args.fertile
    # li_enrich   = parser.parse_args().lithium
    print_xml   = args.print_xml
    run_openmc  = args.run_openmc
    run_debug   = args.run_debug
    n_particles = args.n_particles
    n_cycles    = args.n_cycles


    if run_type == "plot":
        for blanket in blankets:
            for fertile_isotope in isotopes:
                for fertile_kgm3 in densities:
                    current_run = Pile(pile_code=pile_code,
                                         breeder=blanket,
                                 fertile_isotope=fertile_isotope,
                                    fertile_kgm3=fertile_kgm3,
                                        run_type="plot",
                                     n_particles=n_particles,
                                        n_cycles=n_cycles,
                                      run_openmc=run_openmc,)
                    current_run.plot()


    elif run_type == 'tallies':

        for blanket in blankets:  # ['FLiBe', 'DCLL', 'HCPB', 'ARC', 'ARCB']  # make each blanket str match class name
            for fertile_isotope in isotopes:
                for fertile_kgm3 in densities: 
                    current_run = Pile(pile_code=pile_code,
                                       breeder=blanket,
                                       fertile_isotope=fertile_isotope,
                                       fertile_kgm3=fertile_kgm3, 
                                       run_type='tallies',
                                       n_particles=n_particles,
                                       n_cycles=n_cycles,
                                       print_xml=print_xml,
                                       run_openmc=run_openmc,
                                       run_debug=run_debug,
                                       exit_pixel_cm=args.exit_pixel_cm,
                                       exit_energy_bins=args.exit_energy_bins,)

                    print(f"Check if '{current_run.path}' exists: {os.path.isdir(current_run.path)}")

                    current_run.compile()
                    if pile_code == 'a02':
                        from python.gamma_saturation import run_saturation
                        chain_file = args.chain_file or str(
                            Path(current_run.materials.cross_sections).parent.parent / "chain_endfb81_fast.xml")
                        run_saturation(current_run, chain_file, args.gamma_energy_bins)
                        continue
                    statepoint = current_run.openmc()
                    if statepoint is not None:
                        current_run.extract_tallies(statepoint)
                    
                print(f"Collating tallies for {blanket} {fertile_isotope} at {current_run.temp_k}")
                # collate_tallies(blanket, fertile_isotope, current_run.breeder_enrich, current_run.temp_k, current_run.breeder_volume)




if __name__ == '__main__':
    main()

