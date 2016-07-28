from functools import wraps

from alembic.migration import MigrationContext
from alembic.operations import Operations
import sqlalchemy as sa
from toolz.curried import do, operator as op

from zipline.assets.asset_writer import write_version_info
from zipline.errors import AssetDBImpossibleDowngrade


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
    conn = engine.connect()
    metadata = sa.MetaData(conn)
    metadata.reflect(bind=engine)
    version_info_table = metadata.tables['version_info']
    starting_version = sa.select((version_info_table.c.version,)).scalar()

    # Check for accidental upgrade
    if starting_version < desired_version:
        raise AssetDBImpossibleDowngrade(db_version=starting_version,
                                         desired_version=desired_version)

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
        _downgrade_methods[downgrade_key](op, version_info_table)

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
    connection.execute("PRAGMA foreign_keys=%s" % ("ON" if on else "OFF"))


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

        @do(op.setitem(_downgrade_methods, destination))
        @wraps(f)
        def wrapper(op, version_info_table):
            version_info_table.delete().execute()  # clear the version
            f(op)
            write_version_info(version_info_table, destination)

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
    op.drop_index('ix_futures_contracts_root_symbol')
    op.drop_index('ix_futures_contracts_symbol')

    # Execute batch op to allow column modification in SQLite
    with op.batch_alter_table('futures_contracts') as batch_op:

        # Rename 'multiplier'
        batch_op.alter_column(column_name='multiplier',
                              new_column_name='contract_multiplier')

        # Delete 'tick_size'
        batch_op.drop_column('tick_size')

    # Recreate indices after batch
    op.create_index('ix_futures_contracts_root_symbol',
                    table_name='futures_contracts',
                    columns=['root_symbol'])
    op.create_index('ix_futures_contracts_symbol',
                    table_name='futures_contracts',
                    columns=['symbol'],
                    unique=True)


@downgrades(2)
def _downgrade_v2(op):
    """
    Downgrade assets db by removing the 'auto_close_date' column.
    """
    # Drop indices before batch
    # This is to prevent index collision when creating the temp table
    op.drop_index('ix_equities_fuzzy_symbol')
    op.drop_index('ix_equities_company_symbol')

    # Execute batch op to allow column modification in SQLite
    with op.batch_alter_table('equities') as batch_op:

        batch_op.drop_column('auto_close_date')

    # Recreate indices after batch
    op.create_index('ix_equities_fuzzy_symbol',
                    table_name='equities',
                    columns=['fuzzy_symbol'])
    op.create_index('ix_equities_company_symbol',
                    table_name='equities',
                    columns=['company_symbol'])


@downgrades(3)
def _downgrade_v3(op):
    """
    Downgrade assets db by adding a not null constraint on
    ``equities.first_traded``
    """
    op.create_table(
        '_new_equities',
        sa.Column(
            'sid',
            sa.Integer,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column('symbol', sa.Text),
        sa.Column('company_symbol', sa.Text),
        sa.Column('share_class_symbol', sa.Text),
        sa.Column('fuzzy_symbol', sa.Text),
        sa.Column('asset_name', sa.Text),
        sa.Column('start_date', sa.Integer, default=0, nullable=False),
        sa.Column('end_date', sa.Integer, nullable=False),
        sa.Column('first_traded', sa.Integer, nullable=False),
        sa.Column('auto_close_date', sa.Integer),
        sa.Column('exchange', sa.Text),
    )
    op.execute(
        """
        insert into _new_equities
        select * from equities
        where equities.first_traded is not null
        """,
    )
    op.drop_table('equities')
    op.rename_table('_new_equities', 'equities')
    # we need to make sure the indicies have the proper names after the rename
    op.create_index(
        'ix_equities_company_symbol',
        'equities',
        ['company_symbol'],
    )
    op.create_index(
        'ix_equities_fuzzy_symbol',
        'equities',
        ['fuzzy_symbol'],
    )
