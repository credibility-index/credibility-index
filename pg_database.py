import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PostgresDB:
    def __init__(self):
        self.pool = None
        self.init_pool()

    def init_pool(self):
        """Initialize connection pool"""
        try:
            # Используем внутренний хост Railway
            db_host = os.getenv('PGHOST', 'postgres.railway.internal')
            db_user = os.getenv('PGUSER', 'postgres')
            db_password = os.getenv('PGPASSWORD')
            db_port = os.getenv('PGPORT', '5432')
            db_name = os.getenv('POSTGRES_DB', 'railway')

            logger.info(f"Initializing connection pool to {db_host}")

            # Проверяем, установлены ли все необходимые переменные окружения
            if not all([db_host, db_user, db_password, db_port, db_name]):
                raise ValueError("Missing required database connection parameters")

            self.pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=db_host,
                database=db_name,
                user=db_user,
                password=db_password,
                port=db_port,
                cursor_factory=RealDictCursor
            )

            logger.info("PostgreSQL connection pool initialized successfully")
            self.initialize_schema()

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {str(e)}")
            raise

    def initialize_schema(self):
        """Initialize database schema"""
        conn = None
        try:
            conn = self.get_conn()
            with conn.cursor() as cur:
                # Проверяем существование схемы
                cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'media_credibility'")
                if not cur.fetchone():
                    logger.info("Creating schema media_credibility")
                    cur.execute("CREATE SCHEMA media_credibility")

                # Проверяем существование таблиц
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'media_credibility'
                """)
                tables = cur.fetchall()

                if not tables:
                    logger.info("Creating database tables")
                    # Создаем таблицу source_stats
                    cur.execute("""
                        CREATE TABLE media_credibility.source_stats (
                            source TEXT PRIMARY KEY,
                            high INTEGER DEFAULT 0,
                            medium INTEGER DEFAULT 0,
                            low INTEGER DEFAULT 0,
                            total_analyzed INTEGER DEFAULT 0
                        )
                    """)

                    # Добавляем тестовые данные
                    test_data = [
                        ('bbc.com', 15, 5, 1, 21),
                        ('reuters.com', 20, 3, 0, 23),
                        ('foxnews.com', 3, 7, 15, 25),
                        ('cnn.com', 5, 10, 5, 20),
                        ('nytimes.com', 10, 5, 2, 17),
                        ('theguardian.com', 12, 4, 1, 17),
                        ('apnews.com', 18, 2, 0, 20)
                    ]

                    for data in test_data:
                        cur.execute("""
                            INSERT INTO media_credibility.source_stats
                            (source, high, medium, low, total_analyzed)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (source) DO UPDATE
                            SET high = EXCLUDED.high,
                                medium = EXCLUDED.medium,
                                low = EXCLUDED.low,
                                total_analyzed = EXCLUDED.total_analyzed
                        """, data)

            conn.commit()
            logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Error creating schema: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.release_conn(conn)

    def get_conn(self):
        """Get connection from pool"""
        try:
            return self.pool.getconn()
        except Exception as e:
            logger.error(f"Error getting connection: {str(e)}")
            raise

    def release_conn(self, conn):
        """Release connection back to pool"""
        try:
            if conn:
                self.pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error releasing connection: {str(e)}")
            try:
                conn.close()
            except:
                pass

    def close_all_connections(self):
        """Close all connections in the pool"""
        if self.pool:
            self.pool.closeall()
