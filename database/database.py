import os

#import null as null
import psycopg2 as psycopg2
from configparser import ConfigParser
from database import ssh_vars as ssh
from sshtunnel import SSHTunnelForwarder

class database:
    db = None
    tunnel = None

    # db vars
    __select = []
    __where = "1=1"
    __from = ""
    __result = {}

    def __init__(self):
        self.connect_e()


    def config(self, file_name='database.ini', section='postgresql'):
        """
        Receive database informations from database.ini
        :author A.GEZEK
        :date 24.11.2019
        :param string file_name:
        :param string section:
        :return: array -> db connection data
        """
        file_name = os.path.dirname(__file__) + "/" + file_name
        # create a parser
        parser = ConfigParser()
        # read config file
        parser.read(file_name)

        # get section, default to postgresql
        db = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, file_name))

        return db

    def connect(self):
        """
        Connect to database
        :author A.GEZEK
        :date 24.11.2019
        :param: none
        :return void
        """
        conn = None
        try:
            # read connection parameters
            params = self.config()
            # start tunnel because of vagrant
            self.tunnel = SSHTunnelForwarder(
                (ssh.SSH_ADDRESS, ssh.SSH_PORT),
                ssh_username='vagrant',
                ssh_password='vagrant',
                remote_bind_address=('localhost', 5432),
                local_bind_address=('localhost', 5432),  # could be any available port
            )
            self.tunnel.start()

            # connect to the PostgreSQL server
            conn = psycopg2.connect(**params)

            # create a cursor
            self.conn = conn
            self.db = conn.cursor()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def connect_e(self):
        """
              Connect to database
              :author E.Sevim
              :date 15.02.2020
              :param: none
              :return void
              """
        conn = None
        try:
            # read connection parameters
            params = self.config()

            # connect to the PostgreSQL server
            conn = psycopg2.connect(**params)
            # create a cursor
            self.db = conn.cursor()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def close_db(self):
        """
        close db
        :author A.GEZEK
        :date 24.11.2019
        :param none
        :return void
        """
        try:
           # Stop the tunnel
            self.tunnel.stop()
            # close the communication with the PostgreSQL
            self.db.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)


    def select(self, sel):
        """
        Set query's select part
        :author A.GEZEK
        :date 24.11.2019
        :param string|array sel: db's select part
        :return void
        """
        if isinstance(sel, list):
            for slct in sel:
                self.__select.append(slct)
        else:
            self.__select.append(sel)

    def where(self, whr):
        """
        Set query's where part with 'AND'
        :author A.GEZEK
        :date 24.11.2019
        :param string whr: where part
        :return void:
        """
        self.__where = self.__where + " AND " + whr

    def where_or(self , whr ):
        """
        Set query's where part with 'OR'
        :author A.GEZEK
        :date 24.11.2019
        :param string whr: where part
        :returb void:
        """
        self.__where = self.__where + " OR " + whr

    def table(self, tbl):
        """
        Set query's from part
        :author A.GEZEK
        :date 24.11.2019
        :param tbl:
        :return void:
        """
        self.__from = tbl

    def get_result(self):
        """
        Execute query with self parameters and return result
        :author A.GEZEK
        :date 24.11.2019
        :param none:
        :return array:
        """
        self.__result = {}
        if self.__select == "" :
            print( "EMPTY SELECT" )
            return []
        if self.__from == "":
            print( "EMPTY FROM" )
            return []
        try:
            self.db.execute("SELECT " +
                            " , ".join(self.__select) +
                            " FROM " + self.__from +
                            " WHERE " + self.__where
                            + " ORDER BY id")
            self.__result = self.db.fetchall()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            print("error occured")
        # todo close db fonksiyonunda hata var, çağırınca return dönmüyor ve programı kitliyor
        #finally:
            #self.close_db()

        self.__select = []
        self.__from = ""
        self.__where = ""

        return self.__result

    def insert (self, table_name, title):
        """
        Execute query with self parameters and return result
        :author E.SEVİM
        :date 16.03.2020
        :param none:
        :return array:
        """
        print("INSERT INTO %s (pre_title) VALUES(%s, %s) WHERE %s", (table_name, title, self.__where))
        try:
            self.db.execute("INSERT INTO %s (pre_title) VALUES(%s, %s) WHERE (%s)", (table_name, title, self.__where))
            self.conn.commit() # <- We MUST commit to reflect the inserted data
            self.close_db()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
