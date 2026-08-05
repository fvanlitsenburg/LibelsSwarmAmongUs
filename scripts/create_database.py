"""Create the initial LSAU SQLite database."""

from historical_text_pipeline.db import Base, engine


def main() -> None:
    """Create every currently defined database table."""

    Base.metadata.create_all(engine)

    location = engine.url.database or str(engine.url)
    print(f"Database ready: {location}")


if __name__ == "__main__":
    main()