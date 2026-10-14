import openmc


class Geometry:

    def make_geometry(self):

        pile = self.pile

            if self.case == 'C':
                
                # Rectangular lattice
                centers_23  = lattice_coords(ll_in_23, self.geom['shape_in_per'], pitch_in_23)
                # NB. openmc.model.TRISO(outer_radius, fill, center=(0.0, 0.0, 0.0)) -- each model.TRISO instance IS a cell
                biso_in_23  = [openmc.model.TRISO(0.05, u50, center=c) for c in centers_23]
                c23.fill = openmc.model.create_triso_lattice(biso_in_23, ll_in_23, pitch_in_23, self.geom['shape_in_per'], self.blanket)



        self.cells = [c23]
        self.geometry = openmc.Geometry(openmc.Universe(cells=self.cells))