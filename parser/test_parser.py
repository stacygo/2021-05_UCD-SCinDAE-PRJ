import unittest
import pandas as pd
import numpy as np
import parser


class ParserTestCase(unittest.TestCase):
    def test_true_1996_carlow_ardattin(self):
        expected_file_name = 'input/tests/1996_carlow_ardattin.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/1996/carlow/1996 COUNTY CARLOW ARDATTIN.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_1996_cork_castlelyons_bridesbridge(self):
        expected_file_name = 'input/tests/1996_cork_castlelyons_bridesbridge.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/1996/cork/1996 COUNTY CORK CASTLELYONS-BRIDESBRIDGE.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_1999_clare_quin(self):
        expected_file_name = 'input/tests/1999_clare_quin.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/1999/clare/1999 COUNTY CLARE QUIN.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_1999_clare_whitegate(self):
        expected_file_name = 'input/tests/1999_clare_whitegate.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/1999/clare/1999 COUNTY CLARE WHITEGATE-CLARE.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2000_kilkenny_templeorum(self):
        expected_file_name = 'input/tests/2000_kilkenny_templeorum.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/2000/kilkenny/2000 COUNTY KILKENNY TEMPLEORUM.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2000_mayo_louisburgh(self):
        expected_file_name = 'input/tests/2000_mayo_louisburgh.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/2000/mayo/2000 COUNTY MAYO LOUISBURGH.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2002_cork_eyeries(self):
        expected_file_name = 'input/tests/2002_cork_eyeries.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/2002/cork/2002 COUNTY CORK EYERIES.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2004_corkwest_sherkin_island(self):
        expected_file_name = 'input/tests/2004_corkwest_sherkin_island.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        for i in expected:
            print(expected[i])
        actual_file_name = '../crawler/output/pdfs/2004/cork-west/2004 COUNTY CORK_WEST SHERKIN ISLAND.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        for i in actual:
            print(actual[i])
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2005_limerick_broadford(self):
        expected_file_name = 'input/tests/2005_limerick_broadford.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/2005/limerick/2005 COUNTY LIMERICK BROADFORD LIMK 510.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2013_limerick_nicker(self):
        expected_file_name = 'input/tests/2013_limerick_nicker.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/2013/limerick/2013 County Limerick Nicker 2159.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2013_wexford_duncormick(self):
        expected_file_name = 'input/tests/2013_wexford_duncormick.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/2013/wexford/2013 County Wexford Duncormick 2062.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2014_kerry_kilgarvan(self):
        expected_file_name = 'input/tests/2014_kerry_kilgarvan.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/2014/kerry/2014 County Kerry Kilgarvan 970.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2014_kerry_knockanure(self):
        expected_file_name = 'input/tests/2014_kerry_knockanure.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/2014/kerry/2014 County Kerry Knockanure 1149.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2014_kerry_chapeltown(self):
        expected_file_name = 'input/tests/2014_kerry_chapeltown.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/2014/kerry/2014 County Kerry Chapeltown 331.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2018_carlow_tullow(self):
        expected_file_name = 'input/tests/2018_carlow_tullow.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/2018/carlow/2018-County-Carlow-Tullow-48.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)

    def test_true_2019_carlow_clonegal(self):
        expected_file_name = 'input/tests/2019_carlow_clonegal.csv'
        expected = pd.read_csv(expected_file_name,
                               parse_dates=['date'],
                               dtype={'mark': np.float64, 'max_mark': np.float64, 'year': np.float64})
        actual_file_name = '../crawler/output/pdfs/2019/carlow/2019-County-Carlow-Clonegal-37.pdf'
        actual = parser.parse_pdf_to_marks(actual_file_name)
        pd.testing.assert_frame_equal(expected, actual)
        # self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
