from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from db import models
from db.models import base

class LpsOrmConnector:
    MYSQL_PARAMS = {
        'DBTYPE': 'mysql+pymysql',
        'HOST': '10.1.70.113',
        'USER': 'jplee031107',
        'PW': '$kpc1004',
        'DB': 'lps',
        'PORT': 3306
    }

    def __init__(self):
        self.db_url = "".join(
            [self.MYSQL_PARAMS['DBTYPE'], '://',
             self.MYSQL_PARAMS['USER'], ':',
             self.MYSQL_PARAMS['PW'], '@',
             self.MYSQL_PARAMS['HOST'], ':',
             str(self.MYSQL_PARAMS['PORT']), '/',
                 self.MYSQL_PARAMS['DB'], '?charset=utf8'])
        self._session = None
        self._engine = None

    @property
    def session(self):
        return self._session

    def connect(self):
        self.create_engine()
        self.bind_session()
        base.query = self._session.query_property()
        self.reflect()
        return self

    def create_engine(self):
        self._engine = create_engine(self.db_url, convert_unicode=False)
        return self

    def bind_session(self):
        self._session = scoped_session(sessionmaker(autocommit=False,
                                                   autoflush=False,
                                                   bind=self._engine))
        return self

    def reflect(self):
        base.metadata.create_all(self._engine)
        return self

db_session = LpsOrmConnector().connect().session

