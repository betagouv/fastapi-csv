import os
from typing import Union, Type
from pathlib import Path
import inspect
import logging

from fastapi import FastAPI
import fastapi
import pandas as pd
import sqlite3
import numpy as np
import pydantic
import typer
import uvicorn


def create_query_param(name: str, type_: Type, default) -> pydantic.fields.ModelField:
    """Create a query parameter just like fastapi does."""
    param = inspect.Parameter(
        name=name,
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=default,
        annotation=type_,
    )
    field = fastapi.dependencies.utils.get_param_field(
        param=param, param_name=name, default_field_info=fastapi.params.Query
    )
    return field


def dtype_to_type(dtype) -> Type:
    """Convert numpy/pandas dtype to normal Python type."""
    if dtype == np.object:
        return str
    else:
        return type(np.zeros(1, dtype).item())


class FastAPI_CSV(FastAPI):
    def __init__(self, csv_path: Union[str, Path]) -> None:
        """Initializes a FastAPI instance that serves data from a CSV file."""
        super().__init__()

        # Read CSV file and create sqlite3 database from it.
        self.csv_path = Path(csv_path)
        self.table_name = self.csv_path.stem
        self.con = None
        df = self.update_data()

        # Add an endpoint for the CSV file with one query parameter for each column.
        # We hack into fastapi a bit here to inject the query parameters at runtime
        # based on the column names/types.
        def generic_get(**kwargs):
            where_clauses = []
            for name, val in kwargs.items():
                if val is not None:
                    if isinstance(val, str):
                        val = f"'{val}'"
                    where_clauses.append(f"{name}={val}")
            if where_clauses:
                where = "WHERE " + " AND ".join(where_clauses)
            else:
                where = ""

            sql_query = f"SELECT * FROM {self.table_name} {where}"
            print(sql_query)
            cur = self.con.execute(sql_query)
            dicts = cur.fetchall()
            return dicts

        self.get(f"/{self.table_name}", name=self.table_name)(generic_get)
        self._clear_query_params(f"/{self.table_name}")
        for col, dtype in zip(df.columns, df.dtypes):
            type_ = dtype_to_type(dtype)
            self._add_query_param(f"/{self.table_name}", col, type_)

    def delete_data(self):
        """
        Deletes the database with all data read from the CSV. 
            
        The CSV file is not deleted of course. The API endpoints are also not affected,
        so you can use `update_data` to read in new data.
        """
        # See https://stackoverflow.com/questions/48732439/deleting-a-database-file-in-memory
        if self.con is not None:
            logging.info("Deleting old database...")
            # Closing will delete the database, as it's only stored in memory.
            self.con.close()

    def update_data(self):
        """
        Updates the database with the current data from the CSV file.
        
        Note that this only affects the database, not the endpoints. If the column names
        and/or data types in the CSV change (and you want that to update in the 
        endpoints as well), you need to create a new FastAPI_CSV object.
        """
        self.delete_data()

        # Create in-memory sqlite3 database.
        # We can use check_same_thread because we only read from the database, so
        # there's no concurrency
        logging.info("Creating new database...")
        self.con = sqlite3.connect(":memory:", check_same_thread=False)

        # Download excel file from Google Sheets, read it with pandas and write to
        # database.
        df = pd.read_csv(self.csv_path)
        self.con = sqlite3.connect(":memory:", check_same_thread=False)
        df.to_sql(self.table_name, self.con)

        # Make database return dicts instead of tuples.
        # From: https://stackoverflow.com/questions/3300464/how-can-i-get-dict-from-sqlite-query
        def dict_factory(cursor, row):
            d = {}
            for idx, col in enumerate(cursor.description):
                d[col[0]] = row[idx]
            return d

        self.con.row_factory = dict_factory
        logging.info("Database successfully updated")

        return df

    def _find_route(self, route_path):
        """Find a route (stored in the FastAPI instace) by its path (e.g. '/index')."""
        for route in self.router.routes:
            if route.path == route_path:
                return route

    def _clear_query_params(self, route_path):
        """Remove all query parameters of a route."""
        route = self._find_route(route_path)
        # print("Before:", route.dependant.query_params)
        route.dependant.query_params = []
        # print("After:", route.dependant.query_params)

    def _add_query_param(self, route_path, name, type_, default=None):
        """Add a new query parameter to a route."""
        route = self._find_route(route_path)
        # print("Before:", route.dependant.query_params)
        query_param = create_query_param(name, type_, default)
        route.dependant.query_params.append(query_param)
        # print("After:", route.dependant.query_params)


def main(csv_path: str, host: str = "127.0.0.1", port: int = 8000):
    typer.echo(f"🚧 Creating API from CSV file: {csv_path}")
    app = FastAPI_CSV(csv_path)
    typer.echo("🦄 Starting with uvicorn...")
    typer.echo(
        "💡 Check out the API docs at "
        + typer.style(f"https://{host}:{port}/docs", bold=True)
    )
    typer.echo("-" * 80)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    typer.run(main)
