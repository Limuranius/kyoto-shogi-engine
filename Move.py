from Figure import Figure, FigureType


class Move:
    figure: Figure

    # (y, x), 0 <= x, y <= 4
    # Array coordinate system (origin in upper left corner)
    array_destination: tuple[int, int]
    array_origin: tuple[int, int] | None

    # (x, y), 1 <= x, y <= 5
    # Board coordinate system (origin in upper right corner)
    destination: tuple[int, int]
    origin: tuple[int, int] | None

    is_drop: bool

    def __init__(
            self,
            array_destination: tuple[int, int],  # (y, x)
            figure: Figure,
            array_origin: tuple[int, int] = None,  # (y, x)
            is_drop: bool = False,
    ):
        self.array_destination = array_destination
        self.destination = (5 - array_destination[1], array_destination[0] + 1)
        self.array_origin = array_origin
        if array_origin is not None:
            self.origin = (5 - array_origin[1], array_origin[0] + 1)
        else:
            self.origin = None
        self.figure = figure
        self.is_drop = is_drop

    def __str__(self):
        jp = self.figure.type.to_jp()[self.figure.state]
        if self.is_drop:
            return f"{jp} -> {self.array_destination}"
        else:
            return f"{jp} {self.array_origin} -> {self.array_destination}"