import openmc

from python.parameters import *
from python.utilities  import *

class Geometry:

    def build_geometry(self):

        face_length, pile_depth = 20.0, 50.0 # [cm]
        pile_volume = face_length**2 * pile_depth # [cm³]
        hl = face_length / 2.0

        n_triso = round(self.fertile_kgm3 * (pile_volume/1e3) / FERTILE_MASS_IN_TRISO[self.fertile_isotope])  # [#] number of TRISO particles in pile

        # Create TRISO universe
        spheres = [openmc.Sphere(r=r) for r in TRISO_RADII]
        cells   = [openmc.Cell(fill=self.m_xo2,  region=-spheres[0]),
                   openmc.Cell(fill=self.m_buff, region=+spheres[0] & -spheres[1]),
                   openmc.Cell(fill=self.m_PyC1, region=+spheres[1] & -spheres[2]),
                   openmc.Cell(fill=self.m_SiC,  region=+spheres[2] & -spheres[3]),
                   openmc.Cell(fill=self.m_PyC2, region=+spheres[3])]
        u_triso = openmc.Universe(cells=cells)

        boundary = openmc.model.RectangularParallelepiped(-hl, hl, -hl, hl, 0, pile_depth, boundary_type="vacuum")
        box = openmc.Cell(region=-boundary, fill=self.m_breeder)
        lower_left, upper_right = box.region.bounding_box
        self.pile_boundary = boundary
        lengths = upper_right - lower_left
        self.plot_origin = (lower_left + upper_right) / 2

        if n_triso > 0:
            
            shape = np.maximum(1, np.floor(lengths * np.cbrt(n_triso / pile_volume)).astype(int),)
            
            while np.prod(shape) < n_triso:
                shape[np.argmax(lengths / shape)] += 1

            pitch = lengths / shape
            
            if np.any(pitch < 2 * TRISO_RADII[-1]):
                raise ValueError("TRISO diameter exceeds lattice spacing.")

            centers = lattice_coords(lower_left, shape, pitch)

            # Select exactly n_triso sites across the grid
            indices = ((np.arange(n_triso) + 0.5) * len(centers) / n_triso).astype(int)
            trisos = [openmc.model.TRISO(TRISO_RADII[-1], u_triso, center=centers[i]) for i in indices]

            self.plot_origin = min((triso.center for triso in trisos), key=lambda point: np.linalg.norm(point - self.plot_origin)).copy()

            box_shape = (face_length, face_length, pile_depth)
            box.fill = openmc.model.create_triso_lattice(trisos, lower_left, lengths / np.array(box_shape), (20, 20, 50), self.m_breeder)

        self.n_trisos = n_triso
        self.cells = [box]
        self.geometry = openmc.Geometry(openmc.Universe(cells=self.cells))
