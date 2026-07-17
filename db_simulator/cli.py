"""Interfaz de linea de comandos del simulador de bases de datos."""

from __future__ import annotations

import argparse
import sys

from .simulator import Simulator
from .spec import SpecError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="db-simulator",
        description="Simula una base de datos tabular a partir de una especificacion YAML/JSON.",
    )
    parser.add_argument("spec", help="Ruta al fichero de especificacion (YAML o JSON).")
    parser.add_argument("-o", "--output", default="output.csv", help="Ruta del CSV de salida (por defecto output.csv).")
    parser.add_argument("-n", "--rows", type=int, default=None, help="Sobrescribe el numero de filas de la especificacion.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Sobrescribe la semilla aleatoria de la especificacion.")
    parser.add_argument(
        "--report", action="store_true", help="Muestra por pantalla un informe comparando objetivos vs. valores obtenidos."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        simulator = Simulator.from_file(args.spec)
        if args.report:
            df, report = simulator.generate_with_report(n_rows=args.rows, seed=args.seed)
            print(report.to_text())
        else:
            df = simulator.generate(n_rows=args.rows, seed=args.seed)
        df.to_csv(args.output, index=False)
        print(f"\nGeneradas {len(df)} filas x {len(df.columns)} columnas -> {args.output}")
    except SpecError as exc:
        print(f"Error en la especificacion: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"No se encontro el fichero: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
