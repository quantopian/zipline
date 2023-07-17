import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from toolz.curried import do, operator

from zipline.assets.asset_writer import write_version_info
from zipline.errors import AssetDBImpossibleDowngrade
from zipline.utils.compat import wraps
from zipline.utils.preprocess import preprocess
from zipline.utils.sqlite_utils import coerce_string_to_eng


def alter_columns(op, name, *columns, **kwargs):
    """Alter columns from a table.

    Parameters
    ----------
    name : str
        The name of the table.
    *columns
        The new columns to have.
    selection_string : str, optional
        The string to use in the selection. If not provided, it will select all
        of the new columns from the old table.

    Notes
    -----
    The columns are passed explicitly because this should only be used in a
    downgrade where ``zipline.assets.asset_db_schema`` could change.
    """
    selection_string = kwargs.pop("selection_string", None)
    if kwargs:
        raise TypeError(
            "alter_columns received extra arguments: %r" % sorted(kwargs),
        )
    if selection_string is None:
        selection_string = ", ".join(column.name for column in columns)

    tmp_name = "_alter_columns_" + name
    op.rename_table(name, tmp_name)

    for column in columns:
        # Clear any indices that already exist on this table, otherwise we will
        # fail to create the table because the indices will already be present.
        # When we create the table below, the indices that we want to preserve
        # will just get recreated.
        for table in (name, tmp_name):
            try:
                op.execute(f"DROP INDEX IF EXISTS ix_{table}_{column.name}")
            except sa.exc.OperationalError:
                pass

    op.create_table(name, *columns)
    op.execute(
        f"INSERT INTO {name} SELECT {selection_string} FROM {tmp_name}",
    )

    if op.impl.dialect.name == "postgresql":
        op.execute(f"ALTER TABLE {tmp_name} DISABLE TRIGGER ALL;")
        op.execute(f"DROP TABLE {tmp_name} CASCADE;")
    else:
        op.drop_table(tmp_name)


@preprocess(engine=coerce_string_to_eng(require_exists=True))
def downgrade(engine, desired_version):
    """Downgrades the assets db at the given engine to the desired version.

    Parameters
    ----------
    engine : Engine
        An SQLAlchemy engine to the assets database.
    desired_version : int
        The desired resulting version for the assets database.
    """

    # Check the version of the db at the engine
    with engine.begin() as conn:
        metadata_obj = sa.MetaData()
        metadata_obj.reflect(conn)
        version_info_table = metadata_obj.tables["version_info"]
        # starting_version = sa.select((version_info_table.c.version,)).scalar()
        starting_version = conn.execute(
            sa.select(version_info_table.c.version)
        ).scalar()

        # Check for accidental upgrade
        if starting_version < desired_version:
            raise AssetDBImpossibleDowngrade(
                db_version=starting_version, desired_version=desired_version
            )

        # Check if the desired version is already the db version
        if starting_version == desired_version:
            # No downgrade needed
            return

        # Create alembic context
        ctx = MigrationContext.configure(conn)
        op = Operations(ctx)

        # Integer keys of downgrades to run
        # E.g.: [5, 4, 3, 2] would downgrade v6 to v2
        downgrade_keys = range(desired_version, starting_version)[::-1]

        # Disable foreign keys until all downgrades are complete
        _pragma_foreign_keys(conn, False)

        # Execute the downgrades in order
        for downgrade_key in downgrade_keys:
            _downgrade_methods[downgrade_key](op, conn, version_info_table)

        # Re-enable foreign keys
        _pragma_foreign_keys(conn, True)


def _pragma_foreign_keys(connection, on):
    """Sets the PRAGMA foreign_keys state of the SQLite database. Disabling
    the pragma allows for batch modification of tables with foreign keys.

    Parameters
    ----------
    connection : Connection
        A SQLAlchemy connection to the db
    on : bool
        If true, PRAGMA foreign_keys will be set to ON. Otherwise, the PRAGMA
        foreign_keys will be set to OFF.
    """
    if connection.engine.name == "sqlite":
        connection.execute(sa.text(f"PRAGMA foreign_keys={'ON' if on else 'OFF'}"))
    # elif connection.engine.name == "postgresql":
    #     connection.execute(
    #         f"SET session_replication_role  = {'origin' if on else 'replica'};"
    #     )


# This dict contains references to downgrade methods that can be applied to an
# assets db. The resulting db's version is the key.
# e.g. The method at key '0' is the downgrade method from v1 to v0
_downgrade_methods = {}


def downgrades(src):
    """Decorator for marking that a method is a downgrade to a version to the
    previous version.

    Parameters
    ----------
    src : int
        The version this downgrades from.

    Returns
    -------
    decorator : callable[(callable) -> callable]
        The decorator to apply.
    """

    def _(f):
        destination = src - 1

        @do(operator.setitem(_downgrade_methods, destination))
        @wraps(f)
        def wrapper(op, conn, version_info_table):
            conn.execute(version_info_table.delete())  # clear the version
            f(op)
            write_version_info(conn, version_info_table, destination)

        return wrapper

    return _


@downgrades(1)
def _downgrade_v1(op):
    """
    Downgrade assets db by removing the 'tick_size' column and renaming the
    'multiplier' column.
    """
    # Drop indices before batch
    # This is to prevent index collision when creating the temp table
    op.drop_index("ix_futures_contracts_root_symbol")
    op.drop_index("ix_futures_contracts_symbol")

    # Execute batch op to allow column modification in SQLite
    with op.batch_alter_table("futures_contracts") as batch_op:
        # Rename 'multiplier'
        batch_op.alter_column(
            column_name="multiplier", new_column_name="contract_multiplier"
        )

        # Delete 'tick_size'
        batch_op.drop_column("tick_size")

    # Recreate indices after batch
    op.create_index(
        "ix_futures_contracts_root_symbol",
        table_name="futures_contracts",
        columns=["root_symbol"],
    )
    op.create_index(
        "ix_futures_contracts_symbol",
        table_name="futures_contracts",
        columns=["symbol"],
        unique=True,
    )


@downgrades(2)
def _downgrade_v2(op):
    """
    Downgrade assets db by removing the 'auto_close_date' column.
    """
    # Drop indices before batch
    # This is to prevent index collision when creating the temp table
    op.drop_index("ix_equities_fuzzy_symbol")
    op.drop_index("ix_equities_company_symbol")

    # Execute batch op to allow column modification in SQLite
    with op.batch_alter_table("equities") as batch_op:
        batch_op.drop_column("auto_close_date")

    # Recreate indices after batch
    op.create_index(
        "ix_equities_fuzzy_symbol", table_name="equities", columns=["fuzzy_symbol"]
    )
    op.create_index(
        "ix_equities_company_symbol", table_name="equities", columns=["company_symbol"]
    )


@downgrades(3)
def _downgrade_v3(op):
    """
    Downgrade assets db by adding a not null constraint on
    ``equities.first_traded``
    """
    op.create_table(
        "_new_equities",
        sa.Column(
            "sid",
            sa.BigInteger,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column("symbol", sa.Text),
        sa.Column("company_symbol", sa.Text),
        sa.Column("share_class_symbol", sa.Text),
        sa.Column("fuzzy_symbol", sa.Text),
        sa.Column("asset_name", sa.Text),
        sa.Column("start_date", sa.BigInteger, default=0, nullable=False),
        sa.Column("end_date", sa.BigInteger, nullable=False),
        sa.Column("first_traded", sa.BigInteger, nullable=False),
        sa.Column("auto_close_date", sa.BigInteger),
        sa.Column("exchange", sa.Text),
    )
    op.execute(
        """
        insert into _new_equities
        select * from equities
        where equities.first_traded is not null
        """,
    )
    op.drop_table("equities")
    op.rename_table("_new_equities", "equities")
    # we need to make sure the indices have the proper names after the rename
    op.create_index(
        "ix_equities_company_symbol",
        "equities",
        ["company_symbol"],
    )
    op.create_index(
        "ix_equities_fuzzy_symbol",
        "equities",
        ["fuzzy_symbol"],
    )


@downgrades(4)
def _downgrade_v4(op):
    """
    Downgrades assets db by copying the `exchange_full` column to `exchange`,
    then dropping the `exchange_full` column.
    """
    op.drop_index("ix_equities_fuzzy_symbol")
    op.drop_index("ix_equities_company_symbol")

    op.execute("UPDATE equities SET exchange = exchange_full")

    with op.batch_alter_table("equities") as batch_op:
        batch_op.drop_column("exchange_full")

    op.create_index(
        "ix_equities_fuzzy_symbol", table_name="equities", columns=["fuzzy_symbol"]
    )
    op.create_index(
        "ix_equities_company_symbol", table_name="equities", columns=["company_symbol"]
    )


@downgrades(5)
def _downgrade_v5(op):
    op.create_table(
        "_new_equities",
        sa.Column(
            "sid",
            sa.BigInteger,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column("symbol", sa.Text),
        sa.Column("company_symbol", sa.Text),
        sa.Column("share_class_symbol", sa.Text),
        sa.Column("fuzzy_symbol", sa.Text),
        sa.Column("asset_name", sa.Text),
        sa.Column("start_date", sa.BigInteger, default=0, nullable=False),
        sa.Column("end_date", sa.BigInteger, nullable=False),
        sa.Column("first_traded", sa.BigInteger),
        sa.Column("auto_close_date", sa.BigInteger),
        sa.Column("exchange", sa.Text),
        sa.Column("exchange_full", sa.Text),
    )

    op.execute(
        """
        INSERT INTO _new_equities
        SELECT
            equities.sid as sid,
            sym.symbol as symbol,
            sym.company_symbol as company_symbol,
            sym.share_class_symbol as share_class_symbol,
            sym.company_symbol || sym.share_class_symbol as fuzzy_symbol,
            equities.asset_name as asset_name,
            equities.start_date as start_date,
            equities.end_date as end_date,
            equities.first_traded as first_traded,
            equities.auto_close_date as auto_close_date,
            equities.exchange as exchange,
            equities.exchange_full as exchange_full
        FROM
            equities
        INNER JOIN
            -- Select the last held symbol (end_date) for each equity sid from the
            (SELECT
                sid, symbol, company_symbol, share_class_symbol, end_date
                FROM (SELECT *, RANK() OVER (PARTITION BY sid ORDER BY end_date DESC) max_end_date
                FROM equity_symbol_mappings) ranked WHERE max_end_date=1
            ) as sym
        on
            equities.sid = sym.sid
        """,
    )
    op.drop_table("equity_symbol_mappings")
    op.drop_table("equities")
    op.rename_table("_new_equities", "equities")
    # we need to make sure the indices have the proper names after the rename
    op.create_index(
        "ix_equities_company_symbol",
        "equities",
        ["company_symbol"],
    )
    op.create_index(
        "ix_equities_fuzzy_symbol",
        "equities",
        ["fuzzy_symbol"],
    )


@downgrades(6)
def _downgrade_v6(op):
    op.drop_table("equity_supplementary_mappings")


@downgrades(7)
def _downgrade_v7(op):
    tmp_name = "_new_equities"
    op.create_table(
        tmp_name,
        sa.Column(
            "sid",
            sa.BigInteger,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column("asset_name", sa.Text),
        sa.Column("start_date", sa.BigInteger, default=0, nullable=False),
        sa.Column("end_date", sa.BigInteger, nullable=False),
        sa.Column("first_traded", sa.BigInteger),
        sa.Column("auto_close_date", sa.BigInteger),
        # remove foreign key to exchange
        sa.Column("exchange", sa.Text),
        # add back exchange full column
        sa.Column("exchange_full", sa.Text),
    )
    op.execute(
        f"""
        insert into
            {tmp_name}
        select
            eq.sid,
            eq.asset_name,
            eq.start_date,
            eq.end_date,
            eq.first_traded,
            eq.auto_close_date,
            ex.canonical_name,
            ex.exchange
        from
            equities eq
        inner join
            exchanges ex
        on
            eq.exchange = ex.exchange
        where
            ex.country_code in ('US', '??')
        """,
    )
    # if op.impl.dialect.name == "postgresql":
    #     for table_name, col_name in [
    #         ("equities", "exchange"),
    #         ("equity_symbol_mappings", "sid"),
    #         ("equity_supplementary_mappings", "sid"),
    #     ]:
    #         op.drop_constraint(
    #             f"{table_name}_{col_name}_fkey",
    #             f"{table_name}",
    #             type_="foreignkey",
    #         )
    if op.impl.dialect.name == "postgresql":
        op.execute("ALTER TABLE equities DISABLE TRIGGER ALL;")
        op.execute("DROP TABLE equities CASCADE;")
    else:
        op.drop_table("equities")
    op.rename_table(tmp_name, "equities")

    # rebuild all tables without a foreign key to ``exchanges``
    alter_columns(
        op,
        "futures_root_symbols",
        sa.Column(
            "root_symbol",
            sa.Text,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column("root_symbol_id", sa.BigInteger),
        sa.Column("sector", sa.Text),
        sa.Column("description", sa.Text),
        sa.Column("exchange", sa.Text),
    )
    alter_columns(
        op,
        "futures_contracts",
        sa.Column(
            "sid",
            sa.BigInteger,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column("symbol", sa.Text, unique=True, index=True),
        sa.Column("root_symbol", sa.Text, index=True),
        sa.Column("asset_name", sa.Text),
        sa.Column("start_date", sa.BigInteger, default=0, nullable=False),
        sa.Column("end_date", sa.BigInteger, nullable=False),
        sa.Column("first_traded", sa.BigInteger),
        sa.Column("exchange", sa.Text),
        sa.Column("notice_date", sa.BigInteger, nullable=False),
        sa.Column("expiration_date", sa.BigInteger, nullable=False),
        sa.Column("auto_close_date", sa.BigInteger, nullable=False),
        sa.Column("multiplier", sa.Float),
        sa.Column("tick_size", sa.Float),
    )

    # drop the ``country_code`` and ``canonical_name`` columns
    alter_columns(
        op,
        "exchanges",
        sa.Column(
            "exchange",
            sa.Text,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column("timezone", sa.Text),
        # Set the timezone to NULL because we don't know what it was before.
        # Nothing in zipline reads the timezone so it doesn't matter.
        selection_string="exchange, NULL",
    )
    op.rename_table("exchanges", "futures_exchanges")

    # add back the foreign keys that previously existed
    alter_columns(
        op,
        "futures_root_symbols",
        sa.Column(
            "root_symbol",
            sa.Text,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column("root_symbol_id", sa.BigInteger),
        sa.Column("sector", sa.Text),
        sa.Column("description", sa.Text),
        sa.Column(
            "exchange",
            sa.Text,
            sa.ForeignKey("futures_exchanges.exchange"),
        ),
    )
    alter_columns(
        op,
        "futures_contracts",
        sa.Column(
            "sid",
            sa.BigInteger,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column("symbol", sa.Text, unique=True, index=True),
        sa.Column(
            "root_symbol",
            sa.Text,
            sa.ForeignKey("futures_root_symbols.root_symbol"),
            index=True,
        ),
        sa.Column("asset_name", sa.Text),
        sa.Column("start_date", sa.BigInteger, default=0, nullable=False),
        sa.Column("end_date", sa.BigInteger, nullable=False),
        sa.Column("first_traded", sa.BigInteger),
        sa.Column(
            "exchange",
            sa.Text,
            sa.ForeignKey("futures_exchanges.exchange"),
        ),
        sa.Column("notice_date", sa.BigInteger, nullable=False),
        sa.Column("expiration_date", sa.BigInteger, nullable=False),
        sa.Column("auto_close_date", sa.BigInteger, nullable=False),
        sa.Column("multiplier", sa.Float),
        sa.Column("tick_size", sa.Float),
    )

    # Delete equity_symbol_mappings records that no longer refer to valid sids.
    op.execute(
        """
        DELETE FROM
            equity_symbol_mappings
        WHERE
            sid NOT IN (SELECT sid FROM equities);
        """
    )

    # Delete asset_router records that no longer refer to valid sids.
    op.execute(
        """
        DELETE FROM
            asset_router
        WHERE
            sid
            NOT IN (
                SELECT sid FROM equities
                UNION
                SELECT sid FROM futures_contracts
            );
        """
    )
