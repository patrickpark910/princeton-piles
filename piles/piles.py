from pathlib import Path
import argparse

from python.model      import *
from python.parameters import *
from python.utilities  import *

ROOT = Path(__file__).resolve().parent  # resolve() converts to absolute path, parent goes one above

def main():

    (ROOT/"piles"/"figures").mkdir(parents=True, exist_ok=True) 
    (ROOT/"piles"/"openmc").mkdir(parents=True, exist_ok=True) 

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

    # Set defaults for the 'dest' targets above so they are True unless the flag is used
    parser.set_defaults(print_xml=True, run_openmc=True, run_debug=True)

    run_type    = parser.parse_args().run_type
    pile_code   = parser.parse_args().pile_code
    blankets    = parser.parse_args().blankets   # [b.upper() if isinstance(b, str) else b for b in args.blankets]  # too extra to make exception for FLiBe, just type it right yourself :D --ppark
    isotopes    = parser.parse_args().isotopes
    densities   = parser.parse_args().fertile
    li_enrich   = parser.parse_args().lithium
    print_xml   = parser.parse_args().print_xml
    run_openmc  = parser.parse_args().run_openmc 
    run_debug   = parser.parse_args().run_debug
    n_particles = parser.parse_args().n_particles
    n_cycles    = parser.parse_args().n_cycles


    if run_type == 'plot':
        for blanket in blankets:  # make this match class name
            
            current_run = build_pile(blanket, blanket_name=blanket, run_type='plot')

            if os.path.exists(f"{current_run.path}/{current_run.blanket_name}_xz.ppm"):
                print(f"{C.YELLOW}Warning.{C.END} File {current_run.path}/{current_run.blanket_name}_xz.ppm already exists, so this OpenMC volume calculation will be skipped...")
            elif current_run.run_openmc:
                current_run.plot()


    elif run_type == 'tallies':

        for blanket in blankets:  # ['FLiBe', 'DCLL', 'HCPB', 'ARC', 'ARCB']  # make each blanket str match class name
            for fertile_isotope in isotopes:
                for fertile_kgm3 in densities: 
                    current_run = build_pile(blanket,
                                             pile_code=pile_code,
                                             blanket_name=blanket,
                                             fertile_isotope=fertile_isotope,
                                             fertile_kgm3=fertile_kgm3, 
                                             lithium_enrich=li_enrich,
                                             run_type='tallies',
                                             n_particles=n_particles,
                                             n_cycles=n_cycles,
                                             print_xml=print_xml,
                                             run_openmc=run_openmc,
                                             run_debug=run_debug)

                    print(f"Check if '{current_run.path}' exists: {os.path.isdir(current_run.path)}")

                    current_run.compile()
                    current_run.openmc()
                    current_run.extract_tallies()
                    
                print(f"Collating tallies for {blanket} {fertile_isotope} at {current_run.temp_k}")
                # collate_tallies(blanket, fertile_isotope, current_run.breeder_enrich, current_run.temp_k, current_run.breeder_volume)


def build_pile(blanket:str, **kwargs):
    """ Create pile model from mapping into abstract base class (ABC) """
    try:
        cls = eval(blanket) 
        pile = cls(**kwargs)
        pile.initialize()
    except KeyError:
        raise ValueError(f"Unknown blanket '{blanket}'.")
    return pile


if __name__ == '__main__':
    main()

