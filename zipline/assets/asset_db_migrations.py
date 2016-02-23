import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

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


def _downgrade_v1_to_v0(op, version_info_table):
    """
    Downgrade assets db by removing the 'tick_size' column and renaming the
    'multiplier' column.
    """
    version_info_table.delete().execute()

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

    write_version_info(version_info_table, 0)


def _downgrade_v2_to_v1(op, version_info_table):
    """
    Downgrade assets db by removing the 'auto_close_date' column.
    """
    version_info_table.delete().execute()

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

    write_version_info(version_info_table, 1)

# This dict contains references to downgrade methods that can be applied to an
# assets db. The resulting db's version is the key.
# e.g. The method at key '0' is the downgrade method from v1 to v0
_downgrade_methods = {
    0: _downgrade_v1_to_v0,
    1: _downgrade_v2_to_v1,
}
