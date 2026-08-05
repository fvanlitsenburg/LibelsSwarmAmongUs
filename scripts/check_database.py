"""Verify connections to the development and test databases."""

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from historical_text_pipeline.config.settings import get_settings
from historical_text_pipeline.db.session import make_engine


def check_connection(label: str, database_url: str) -> bool:
    """Connect to one database and print its identity."""

    engine = make_engine(database_url)

    try:
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT
                        current_database() AS database_name,
                        current_user AS user_name,
                        version() AS postgres_version
                    """
                )
            ).mappings().one()

        print(f"{label}: OK")
        print(f"  Database: {row['database_name']}")
        print(f"  User:     {row['user_name']}")
        print(f"  Version:  {row['postgres_version']}")
        return True

    except SQLAlchemyError as error:
        print(f"{label}: FAILED")
        print(f"  {error}")
        return False

    finally:
        engine.dispose()


def main() -> None:
    """Check both configured databases."""

    settings = get_settings()

    development_ok = check_connection(
        "Development database",
        settings.database_url,
    )

    test_ok = check_connection(
        "Test database",
        settings.test_database_url,
    )

    if not development_ok or not test_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()