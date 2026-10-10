import gaiya
from gaiya import Plane, Property

# the ARC field: a flow plane wrapping r, c, cell varieties
arc_field=Plane(
    grid=lambda g: (len(g), len(g[0])),
    properties=[
        Property("colour",
                needs={"subject", "position"},
                measure=lambda ctx: ctx.subject[ctx.position[0]][ctx.position[1]],
                ),
    ],
)
# Test arc grid dummy
test_grid=[[1, 2, 3],
            [4, 5, 6]]
# build psi
psi= arc_field.compose(test_grid)
# to read back
print("shape:", psi.grid)
print("number of cells:", psi.n)
print("colour at row 1, col 2:", psi.cell((1, 2)).properties["colour"])