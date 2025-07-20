"""CLI commands for TestWriter automation generation."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from .parser import MarkdownTestCaseParser
# Import unified document parser
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from testteller.core.data_ingestion.unified_document_parser import UnifiedDocumentParser, DocumentType
from .generators import PythonTestGenerator, JavaScriptTestGenerator, JavaTestGenerator, TypeScriptTestGenerator

logger = logging.getLogger(__name__)

# Import supported languages and frameworks from central constants
from testteller.core.constants import SUPPORTED_LANGUAGES, SUPPORTED_FRAMEWORKS


def get_generator(language: str, framework: str, output_dir: Path):
    """Get the appropriate generator based on language and framework."""
    if language == 'python':
        return PythonTestGenerator(framework, output_dir)
    elif language == 'javascript':
        return JavaScriptTestGenerator(framework, output_dir)
    elif language == 'typescript':
        return TypeScriptTestGenerator(framework, output_dir)
    elif language == 'java':
        return JavaTestGenerator(framework, output_dir)
    else:
        raise ValueError(f"Unsupported language: {language}")


def validate_framework(language: str, framework: str) -> bool:
    """Validate that the framework is supported for the language."""
    return framework in SUPPORTED_FRAMEWORKS.get(language, [])


def automate_command(
    input_file: Annotated[str, typer.Argument(help="Path to test cases file (supports .md, .txt, .pdf, .docx, .xlsx)")],
    language: Annotated[str, typer.Option(
        "--language", "-l", help="Programming language for test automation")] = None,
    framework: Annotated[str, typer.Option(
        "--framework", "-F", help="Test framework to use")] = None,
    output_dir: Annotated[str, typer.Option(
        "--output-dir", "-o", help="Output directory for generated tests")] = "./generated_tests",
    interactive: Annotated[bool, typer.Option(
        "--interactive", "-i", help="Interactive mode to select test cases")] = False,
    enhance: Annotated[bool, typer.Option(
        "--enhance", "-E", help="Use LLM to enhance generated test code")] = False,
    llm_provider: Annotated[str, typer.Option(
        "--llm-provider", "-p", help="LLM provider for enhancement (gemini, openai, claude, llama)")] = None
):
    """Generate automation code from TestTeller test cases."""
    
    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        print(f"❌ Error: Test cases file '{input_file}' not found.")
        raise typer.Exit(code=1)
    
    # Interactive language/framework selection if not provided
    if not language:
        print("\n🔧 Test Automation Configuration")
        print("================================")
        print("\nSelect programming language:")
        languages = list(SUPPORTED_FRAMEWORKS.keys())
        for i, lang in enumerate(languages, 1):
            print(f"  {i}. {lang}")
        
        while True:
            try:
                choice = typer.prompt("\nEnter number", type=int)
                if 1 <= choice <= len(languages):
                    language = languages[choice - 1]
                    break
                print("Invalid choice. Please try again.")
            except (ValueError, typer.Abort):
                print("Invalid input. Please enter a number.")
    
    if not framework:
        frameworks = SUPPORTED_FRAMEWORKS[language]
        print(f"\nSelect test framework for {language}:")
        for i, fw in enumerate(frameworks, 1):
            print(f"  {i}. {fw}")
        
        while True:
            try:
                choice = typer.prompt("\nEnter number", type=int)
                if 1 <= choice <= len(frameworks):
                    framework = frameworks[choice - 1]
                    break
                print("Invalid choice. Please try again.")
            except (ValueError, typer.Abort):
                print("Invalid input. Please enter a number.")
    
    # Validate framework
    if not validate_framework(language, framework):
        print(f"❌ Error: Framework '{framework}' is not supported for language '{language}'.")
        print(f"Supported frameworks for {language}: {', '.join(SUPPORTED_LANGUAGES[language])}")
        raise typer.Exit(code=1)
    
    print(f"\n✅ Configuration:")
    print(f"  • Language: {language}")
    print(f"  • Framework: {framework}")
    print(f"  • Output: {output_dir}")
    
    # LLM Enhancement Configuration (if not already specified)
    if not enhance:
        try:
            from .llm_enhancer import is_llm_enhancement_available
            
            if is_llm_enhancement_available():
                print(f"\n🤖 LLM Enhancement Available")
                print("=============================")
                print("TestTeller can use AI to enhance the generated test code with:")
                print("  • Improved error handling and edge cases")
                print("  • Better assertions and validations")
                print("  • Enhanced code quality and best practices")
                print("  • Framework-specific optimizations")
                
                enhance = typer.confirm("\nWould you like to enable LLM enhancement?", default=False)
                
                if enhance:
                    if not llm_provider:
                        print("\nAvailable LLM providers: gemini, openai, claude, llama")
                        llm_provider = typer.prompt(
                            "Select LLM provider (leave empty for default)", 
                            default="", 
                            show_default=False
                        ).strip()
                        llm_provider = llm_provider if llm_provider else None
                    
                    print(f"✅ LLM enhancement enabled" + (f" with {llm_provider}" if llm_provider else ""))
                else:
                    print("⏭️  LLM enhancement disabled")
            else:
                print(f"\n💡 Tip: Configure TestTeller with an LLM provider to enable AI-powered test enhancement")
        except ImportError:
            pass  # LLM enhancement not available
        except Exception as e:
            logger.warning(f"Failed to check LLM availability: {e}")
    
    try:
        # Parse test cases using unified parser
        print(f"\n📖 Parsing test cases from: {input_file}")
        file_extension = input_path.suffix.lower()
        
        if file_extension not in ['.md', '.txt', '.pdf', '.docx', '.xlsx']:
            print(f"❌ Unsupported file format: {file_extension}")
            print("Supported formats: .md, .txt, .pdf, .docx, .xlsx")
            raise typer.Exit(code=1)
        
        # Use unified parser for all formats
        print(f"  📄 Detected format: {file_extension}")
        unified_parser = UnifiedDocumentParser()
        
        # Parse document for automation
        parsed_doc = asyncio.run(unified_parser.parse_for_automation(input_path))
        
        # Extract test cases
        test_cases = parsed_doc.test_cases
        
        # If no structured test cases found, try to use content for context
        if not test_cases:
            print(f"  📝 No structured test cases found. Document type: {parsed_doc.metadata.document_type.value}")
            
            if parsed_doc.metadata.document_type in [DocumentType.TEST_CASES, DocumentType.REQUIREMENTS]:
                # For markdown files, try the legacy parser as fallback
                if file_extension == '.md':
                    print("  🔄 Trying markdown-specific parser...")
                    md_parser = MarkdownTestCaseParser()
                    test_cases = md_parser.parse_file(input_path)
                
                if not test_cases:
                    print("❌ No test cases found in the file.")
                    print("💡 Tip: Ensure the file contains structured test cases in the expected format.")
                    print("   For supported formats, see: https://github.com/testteller/docs")
                    raise typer.Exit(code=1)
            else:
                print("❌ This document doesn't appear to contain test cases.")
                print(f"   Detected type: {parsed_doc.metadata.document_type.value}")
                print("💡 Try using a file that contains structured test cases.")
                raise typer.Exit(code=1)
        
        print(f"✅ Found {len(test_cases)} test cases")
        
        # Show document info
        if parsed_doc.metadata.title:
            print(f"  📋 Document: {parsed_doc.metadata.title}")
        if parsed_doc.metadata.sections:
            print(f"  📑 Sections: {len(parsed_doc.metadata.sections)}")
        print(f"  📊 Content: {parsed_doc.metadata.word_count} words, {parsed_doc.metadata.character_count} characters")
        
        # Interactive selection if requested
        if interactive:
            selected_cases = interactive_select_tests(test_cases)
            if not selected_cases:
                print("❌ No test cases selected.")
                raise typer.Exit(code=1)
            test_cases = selected_cases
        
        # Generate automation code
        print(f"\n🚀 Generating {language} automation code...")
        output_path = Path(output_dir)
        generator = get_generator(language, framework, output_path)
        
        generated_files = generator.generate(test_cases)
        
        # LLM Enhancement (optional)
        if enhance:
            print(f"\n🤖 Enhancing generated code with LLM...")
            try:
                from .llm_enhancer import create_enhancer, is_llm_enhancement_available
                
                if not is_llm_enhancement_available():
                    print("⚠️  LLM enhancement not available. Ensure LLM is configured in TestTeller.")
                    print("   Generated code will be used without enhancement.")
                else:
                    enhancer = create_enhancer(provider=llm_provider)
                    if enhancer.is_available():
                        enhanced_files = enhancer.enhance_generated_tests(
                            generated_files, language, framework
                        )
                        
                        # Count enhanced files
                        enhanced_count = sum(1 for filename, content in enhanced_files.items() 
                                           if content != generated_files.get(filename, ""))
                        
                        if enhanced_count > 0:
                            generated_files = enhanced_files
                            print(f"✅ Enhanced {enhanced_count} test files using {enhancer.provider}")
                            
                            # Show optimization suggestions
                            suggestions = enhancer.get_optimization_suggestions(language, framework)
                            if suggestions:
                                print(f"\n💡 Optimization suggestions for {language} + {framework}:")
                                for i, suggestion in enumerate(suggestions[:5], 1):  # Show top 5
                                    print(f"  {i}. {suggestion}")
                        else:
                            print("⚠️  No files were enhanced. Using original generated code.")
                    else:
                        print("⚠️  LLM enhancer initialization failed. Using original generated code.")
                        
            except ImportError:
                print("⚠️  LLM enhancement dependencies not available. Using original generated code.")
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")
                print(f"⚠️  LLM enhancement failed: {e}")
                print("   Using original generated code.")
        
        # Write files
        print(f"\n📝 Writing generated files to: {output_dir}")
        generator.write_files(generated_files)
        
        # Summary
        print(f"\n✅ Successfully generated {len(generated_files)} files:")
        for file_name in generated_files:
            print(f"  • {output_path / file_name}")
        
        # Next steps
        print_next_steps(language, framework, output_path)
        
    except Exception as e:
        logger.error(f"Failed to generate automation code: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        raise typer.Exit(code=1)


def interactive_select_tests(test_cases):
    """Interactive test case selection."""
    print("\n📋 Available test cases:")
    for i, tc in enumerate(test_cases, 1):
        print(f"{i:3d}. [{tc.id}] {tc.objective[:60]}...")
    
    print("\nSelect test cases to automate:")
    print("  • Enter numbers separated by commas (e.g., 1,3,5)")
    print("  • Enter ranges (e.g., 1-5)")
    print("  • Enter 'all' to select all tests")
    print("  • Enter 'none' to cancel")
    
    selection = typer.prompt("\nYour selection").strip().lower()
    
    if selection == 'none':
        return []
    elif selection == 'all':
        return test_cases
    else:
        selected_indices = parse_selection(selection, len(test_cases))
        return [test_cases[i-1] for i in selected_indices]


def parse_selection(selection: str, max_index: int) -> list:
    """Parse user selection string into list of indices."""
    indices = set()
    
    for part in selection.split(','):
        part = part.strip()
        if '-' in part:
            # Range
            try:
                start, end = map(int, part.split('-'))
                indices.update(range(max(1, start), min(max_index + 1, end + 1)))
            except ValueError:
                print(f"⚠️  Invalid range: {part}")
        else:
            # Single number
            try:
                num = int(part)
                if 1 <= num <= max_index:
                    indices.add(num)
                else:
                    print(f"⚠️  Index out of range: {num}")
            except ValueError:
                print(f"⚠️  Invalid number: {part}")
    
    return sorted(indices)


def print_next_steps(language: str, framework: str, output_dir: Path):
    """Print next steps based on language and framework."""
    print("\n📚 Next steps:")
    
    if language == 'python':
        print(f"  1. cd {output_dir}")
        print("  2. pip install -r requirements.txt")
        if framework == 'pytest':
            print("  3. pytest")
        else:
            print("  3. python -m unittest discover")
    
    elif language == 'javascript':
        print(f"  1. cd {output_dir}")
        print("  2. npm install")
        print("  3. npm test")
    
    elif language == 'java':
        print(f"  1. cd {output_dir}")
        print("  2. mvn clean install")
        print("  3. mvn test")
    
    print("\n💡 Tips:")
    print("  • Review and customize the generated tests")
    print("  • Update test data and assertions")
    print("  • Configure test environment settings")
    print("  • Add missing test implementations marked with TODO")