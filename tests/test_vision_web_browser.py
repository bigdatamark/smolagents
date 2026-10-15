# [MODULE DOC]
# Description: Tests for the XPath injection fix in vision_web_browser: _escape_xpath_string quoting rules and
#   search_item_ctrl_f behavior with a mocked Selenium driver.
# Inputs: depends on pytest, unittest.mock and smolagents.vision_web_browser (requires helium/selenium
#   installed).
# Outputs: mock_driver fixture and the test classes.
# Exceptions: none raised directly.
# Side effects: importing the module under test imports helium and selenium.
"""Test XPath injection vulnerability fix in vision_web_browser.py"""

from unittest.mock import Mock, patch

import pytest

from smolagents.vision_web_browser import _escape_xpath_string, search_item_ctrl_f


@pytest.fixture
def mock_driver():
    # [FUNCTION DOC]
    # Description: HELPER: Fixture returning a Mock driver whose find_elements returns one mock element and
    #   whose execute_script returns None.
    # Inputs: none.
    # Outputs: Mock.
    # Exceptions: none raised directly.
    # Side effects: none.
    """Mock Selenium WebDriver"""
    driver = Mock()
    driver.find_elements.return_value = [Mock()]  # Mock found elements
    driver.execute_script.return_value = None
    return driver


class TestXPathEscaping:
    # [CLASS DOC]
    # Description: Tests for _escape_xpath_string.
    # Inputs: none.
    # Outputs: test outcomes.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    """Test XPath string escaping functionality"""

    @pytest.mark.parametrize(
        "input_text,expected_pattern",
        [
            ("normal text", "'normal text'"),
            ("text with 'quote'", "\"text with 'quote'\""),
            ('text with "quote"', "'text with \"quote\"'"),
            ("text with one single'quote", '"text with one single\'quote"'),
            ('text with one double"quote', "'text with one double\"quote'"),
            (
                "text with both 'single' and \"double\" quotes",
                "concat('text with both ', \"'\", 'single', \"'\", ' and \"double\" quotes')",
            ),
            ("", "''"),
            ("'", '"\'"'),
            ('"', "'\"'"),
        ],
    )
    def test_escape_xpath_string_basic(self, input_text, expected_pattern):
        # [FUNCTION DOC]
        # Description: Checks exact escaped output for plain, single-quote, double-quote, mixed, empty and
        #   single-character inputs.
        # Inputs: self; input_text, expected_pattern (parametrized).
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test basic XPath escaping cases"""
        result = _escape_xpath_string(input_text)
        assert result == expected_pattern

    @pytest.mark.parametrize(
        "input_text",
        [
            "text with both 'single' and \"double\" quotes",
            'it\'s a "test" case',
            "'mixed\" quotes'",
        ],
    )
    def test_escape_xpath_string_mixed_quotes(self, input_text):
        # [FUNCTION DOC]
        # Description: Checks strings with both quote kinds produce a concat expression.
        # Inputs: self; input_text (parametrized).
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test XPath escaping with mixed quotes uses concat()"""
        result = _escape_xpath_string(input_text)
        assert result.startswith("concat(")
        assert result.endswith(")")

    @pytest.mark.parametrize(
        "malicious_input",
        [
            "')] | //script[@src='evil.js'] | foo[contains(text(), '",
            "') or 1=1 or ('",
            "')] | //user[contains(@role,'admin')] | foo[contains(text(), '",
            "') and substring(//user[1]/password,1,1)='a",
        ],
    )
    def test_escape_prevents_injection(self, malicious_input):
        # [FUNCTION DOC]
        # Description: Checks injection payloads come out fully quoted or as a concat expression.
        # Inputs: self; malicious_input (parametrized).
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        # KNOWN ISSUE: The assertion only checks the first and last characters, which a naive unescaped
        #   wrapping could also satisfy; it is a weak check.
        """Test that malicious XPath injection attempts are safely escaped"""
        result = _escape_xpath_string(malicious_input)
        # Should either be wrapped in quotes or use concat()
        assert (
            (result.startswith("'") and result.endswith("'"))
            or (result.startswith('"') and result.endswith('"'))
            or result.startswith("concat(")
        )


class TestSearchItemCtrlF:
    # [CLASS DOC]
    # Description: Tests for search_item_ctrl_f using a patched module-level driver.
    # Inputs: none.
    # Outputs: test outcomes.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    """Test the search_item_ctrl_f function with XPath injection protection"""

    @pytest.mark.parametrize(
        "search_text",
        [
            "normal search",
            "search with 'quotes'",
            'search with "quotes"',
            "')] | //script[@src='evil.js'] | foo[contains(text(), '",
            "') or 1=1 or ('",
        ],
    )
    def test_search_item_prevents_injection(self, search_text, mock_driver):
        # [FUNCTION DOC]
        # Description: Patches the module driver, calls the tool, checks find_elements was called once and,
        #   for injection inputs, that the XPath uses concat or contains quotes, and that the result mentions
        #   Found.
        # Inputs: self; search_text (parametrized); mock_driver.
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: temporarily sets the module-global driver.
        # KNOWN ISSUE: The quote-count check is very weak (any XPath here contains at least two quotes).
        """Test that search_item_ctrl_f prevents XPath injection"""
        with patch("smolagents.vision_web_browser.driver", mock_driver, create=True):
            # Call the function
            result = search_item_ctrl_f(search_text)

            # Verify driver.find_elements was called
            mock_driver.find_elements.assert_called_once()

            # Get the actual XPath query that was generated
            call_args = mock_driver.find_elements.call_args
            xpath_query = call_args[0][1]  # Second positional argument

            # Verify the query doesn't contain unescaped injection
            if "')] | //" in search_text:
                # For injection attempts, verify they're properly escaped
                # The query should either use concat() or be properly quoted
                is_concat = "concat(" in xpath_query
                is_properly_quoted = xpath_query.count('"') >= 2 or xpath_query.count("'") >= 2
                assert is_concat or is_properly_quoted, f"XPath injection not prevented: {xpath_query}"

            # Verify we got a result
            assert "Found" in result

    def test_search_item_nth_result(self, mock_driver):
        # [FUNCTION DOC]
        # Description: With three mock matches, checks nth_result 2 reports 3 matches and focus on element 2
        #   of 3.
        # Inputs: self; mock_driver.
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: temporarily sets the module-global driver.
        """Test nth_result parameter works correctly"""
        mock_driver.find_elements.return_value = [Mock(), Mock(), Mock()]  # 3 elements

        with patch("smolagents.vision_web_browser.driver", mock_driver, create=True):
            result = search_item_ctrl_f("test", nth_result=2)

            # Should find 3 matches and focus on element 2
            assert "Found 3 matches" in result
            assert "Focused on element 2 of 3" in result

    def test_search_item_not_found(self, mock_driver):
        # [FUNCTION DOC]
        # Description: With one match, checks nth_result 3 raises an Exception with the not-found message.
        # Inputs: self; mock_driver.
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: temporarily sets the module-global driver.
        """Test exception when nth_result exceeds available matches"""
        mock_driver.find_elements.return_value = [Mock()]  # Only 1 element

        with patch("smolagents.vision_web_browser.driver", mock_driver, create=True):
            with pytest.raises(Exception, match="Match n°3 not found"):
                search_item_ctrl_f("test", nth_result=3)
