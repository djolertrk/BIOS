//===- Lexer.h - Lexer for the bios language ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple Lexer for the bios language.
//
//===----------------------------------------------------------------------===//

#ifndef blang_LEXER_H_
#define blang_LEXER_H_

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <iostream>
#include <cstring>

namespace blang {

/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};

// List of Token returned by the lexer.
enum Token : int {
  tok_semicolon = ';',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',
  tok_colon = ':',
  tok_single_quote = '\'',

  tok_eof = -1,

  // Commands.
  tok_return = -2,
  tok_var = -3,
  tok_fn = -4,

  // Types.
  tok_tensor = -5,
  tok_float = -6,
  tok_float64 = -7,
  tok_int = -8,
  tok_int64 = -9,
  tok_char = -10,
  tok_string = -11,
  tok_struct = -12,

  // Primary.
  tok_identifier = -13,
  tok_number = -14,
  tok_character = -15,
  tok_string_primary = -16,
};

/// The Lexer is an abstract base class providing all the facilities that the
/// Parser expects. It goes through the stream one token at a time and keeps
/// track of the location in the file for debugging purposes.
/// It relies on a subclass to provide a `readNextLine()` method. The subclass
/// can proceed by reading the next line from the standard input or from a
/// memory mapped file.
class Lexer {
public:
  /// Create a lexer for the given filename. The filename is kept only for
  /// debugging purposes (attaching a location to a Token).
  Lexer(std::string filename)
      : lastLocation(
            {std::make_shared<std::string>(std::move(filename)), 0, 0}) {}
  virtual ~Lexer() = default;

  /// Look at the current token in the stream.
  Token getCurToken() { return curTok; }

  /// Move to the next token in the stream and return it.
  Token getNextToken() { return curTok = getTok(); }

  /// Move to the next token in the stream, asserting on the current token
  /// matching the expectation.
  void consume(Token tok) {
    assert(tok == curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  /// Return the current identifier (prereq: getCurToken() == tok_identifier)
  llvm::StringRef getId() {
    assert(curTok == tok_identifier);
    return identifierStr;
  }

  /// Return the current number (prereq: getCurToken() == tok_number)
  double getValue() {
    assert(curTok == tok_number);
    return numVal;
  }

  /// Return the current number (prereq: getCurToken() == tok_character)
  char getCharValue() {
    assert(curTok == tok_character);
    return charVal;
  }

  /// Return the current number (prereq: getCurToken() == tok_string_primary)
  std::string& getStringValue() {
    assert(curTok == tok_string_primary);
    return stringVal;
  }

  /// Return the location for the beginning of the current token.
  Location getLastLocation() { return lastLocation; }

  // Return the current line in the file.
  int getLine() { return curLineNum; }

  // Return the current column in the file.
  int getCol() { return curCol; }

  // Return true if the token represents a type.
  bool isType(Token tok) {
    switch(tok) {
    default:
      return false;
    case tok_float:
    case tok_float64:
    case tok_tensor:
    case tok_int:
    case tok_int64:
    case tok_char:
    case tok_string:
      return true;
    }
  }

private:
  /// Delegate to a derived class fetching the next line. Returns an empty
  /// string to signal end of file (EOF). Lines are expected to always finish
  /// with "\n".
  virtual llvm::StringRef readNextLine() = 0;

  /// Return the next character from the stream. This manages the buffer for the
  /// current line and request the next line buffer to the derived class as
  /// needed.
  int getNextChar() {
    // The current line buffer should not be empty unless it is the end of file.
    if (curLineBuffer.empty())
      return EOF;
    ++curCol;
    auto nextchar = curLineBuffer.front();
    curLineBuffer = curLineBuffer.drop_front();
    if (curLineBuffer.empty())
      curLineBuffer = readNextLine();
    if (nextchar == '\n') {
      ++curLineNum;
      curCol = 0;
    }
    return nextchar;
  }

  ///  Return the next token from standard input.
  Token getTok() {
    // Skip any whitespace.
    while (isspace(lastChar))
      lastChar = Token(getNextChar());

    // Save the current location before reading the token characters.
    lastLocation.line = curLineNum;
    lastLocation.col = curCol;

    // Parse the character from char type.
    if (lastChar == '\'') {
        lastChar = Token(getNextChar());
        charVal = (char)lastChar;
        // Eat the second '\''.
        lastChar = Token(getNextChar());
        if ((char)lastChar == '\'') {
          lastChar = Token(getNextChar());
          return tok_character;
        }
    }

    // Parse the string from string type.
    if (lastChar == '\"') {
        lastChar = Token(getNextChar());
        std::string read_string;
        read_string += (char)lastChar;
        while ((isalnum((lastChar = Token(getNextChar()))) || isspace(lastChar))
               && lastChar != '\"')
         read_string += (char)lastChar;

        stringVal = std::move(read_string);

        assert(lastChar == '\"' && "double quote should close the string");
        lastChar = Token(getNextChar());
        return tok_string_primary;
    }

    // Identifier: [a-zA-Z][a-zA-Z0-9_]*.
    if (isalpha(lastChar)) {
      identifierStr = (char)lastChar;

      while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
        identifierStr += (char)lastChar;

      if (identifierStr == "return")
        return tok_return;
      if (identifierStr == "fn")
        return tok_fn;
      if (identifierStr == "tensor")
        return tok_tensor;
      if (identifierStr == "float")
        return tok_float;
      if (identifierStr == "float64")
        return tok_float64;
      if (identifierStr == "int")
        return tok_int;
      if (identifierStr == "int64")
        return tok_int64;
      if (identifierStr == "char")
        return tok_char;
      if (identifierStr == "string")
        return tok_string;
      if (identifierStr == "struct")
        return tok_struct;
      if (identifierStr == "var")
        return tok_var;
      return tok_identifier;
    }

    // Number: [0-9] ([0-9.])*.
    if (isdigit(lastChar)) {
      std::string numStr;
      do {
        numStr += lastChar;
        lastChar = Token(getNextChar());
      } while (isdigit(lastChar) || lastChar == '.');

      numVal = strtod(numStr.c_str(), nullptr);
      return tok_number;
    }

    // A comment.
    if (lastChar == '#') {
      // Comment until end of line.
      do {
        lastChar = Token(getNextChar());
      } while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

      if (lastChar != EOF)
        return getTok();
    }

    // Check for end of file. Don't eat the EOF.
    if (lastChar == EOF)
      return tok_eof;

    // Otherwise, just return the character as its ascii value.
    Token thisChar = Token(lastChar);
    lastChar = Token(getNextChar());
    return thisChar;
  }

  /// The last token read from the input.
  Token curTok = tok_eof;

  /// Location for \p curTok.
  Location lastLocation;

  /// If the current Token is an identifier, this string contains the value.
  std::string identifierStr;

  /// If the current Token is a number, this contains the value.
  double numVal = 0;

  char charVal = 0;

  std::string stringVal;

  /// The last value returned by getNextChar(). We need to keep it around as we
  /// always need to read ahead one character to decide when to end a token and
  /// we can't put it back in the stream after reading from it.
  Token lastChar = Token(' ');

  /// Keep track of the current line number in the input stream.
  int curLineNum = 0;

  /// Keep track of the current column number in the input stream
  int curCol = 0;

  /// Buffer supplied by the derived class on calls to `readNextLine()`.
  llvm::StringRef curLineBuffer = "\n";
};

/// A lexer implementation operating on a buffer in memory.
class LexerBuffer final : public Lexer {
public:
  LexerBuffer(const char *begin, const char *end, std::string filename)
      : Lexer(std::move(filename)), current(begin), end(end) {}

private:
  /// Provide one line at a time to the Lexer, return an empty string when
  /// reaching the end of the buffer.
  llvm::StringRef readNextLine() override {
    auto *begin = current;
    while (current <= end && *current && *current != '\n')
      ++current;
    if (current <= end && *current)
      ++current;
    llvm::StringRef result{begin, static_cast<size_t>(current - begin)};
    return result;
  }
  const char *current, *end;
};
} // namespace blang

#endif // blang_LEXER_H_
